#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

BASE = "https://api.gbif.org/v1"
UA = "wild-forager-gbif-image-sampler/1.0"

MEDIA_TYPE = "StillImage"

# Paging
DEFAULT_LIMIT = 300

# Be polite
DEFAULT_SLEEP_SEARCH = 0.12
DEFAULT_SLEEP_DOWNLOAD = 0.2
DEFAULT_BATCH_SLEEP = 1.0
DEFAULT_TIMEOUT = 60
DEFAULT_DOWNLOAD_RETRIES = 4
DEFAULT_DOWNLOAD_RETRY_SLEEP = 2.0

# Commercial-safe media licenses only.
COMMERCIAL_LICENSES = {
    "cc0-1.0": "CC0 1.0",
    "cc-by-4.0": "CC BY 4.0",
    "cc-by-sa-4.0": "CC BY-SA 4.0",
    "cc-by-3.0": "CC BY 3.0",
    "cc-by-sa-3.0": "CC BY-SA 3.0",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT.parent
SYNONYMS_JSON_DEFAULT = APP_ROOT / "assets" / "data" / "synonyms.json"


def http_get(
    session: requests.Session,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = 4,
    retry_sleep: float = 2.0,
) -> Dict[str, Any]:
    url = f"{BASE}{path}"
    headers = {"User-Agent": UA}
    for attempt in range(max_retries):
        try:
            r = session.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(retry_sleep * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(retry_sleep * (attempt + 1))
            continue
    r.raise_for_status()
    return r.json()


def safe_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def write_csv(rows: List[Dict[str, Any]], filename: Path) -> None:
    if not rows:
        filename.write_text("")
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with filename.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {}
            for k in fieldnames:
                v = r.get(k, "")
                if isinstance(v, (dict, list)):
                    out[k] = safe_json(v)
                else:
                    out[k] = v
            w.writerow(out)


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


def media_is_commercial_ok(media_obj: Dict[str, Any]) -> Tuple[bool, str, str]:
    lic = media_obj.get("license")
    key = normalize_license_to_key(lic)
    if not key:
        return (False, "", "")
    return (True, key, COMMERCIAL_LICENSES[key])


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

    parts.append(f"License: {license_label}")
    if source:
        parts.append(f"Source: {source}")
    if dataset_key:
        parts.append(f"GBIF dataset: {dataset_key}")
    if occ_key:
        parts.append(f"Occurrence: {occ_key}")
    return " | ".join(parts)


def quality_score(
    occ: Dict[str, Any],
    media_obj: Dict[str, Any],
    license_key: str,
    preferred_country: Optional[str],
) -> int:
    score = 0

    if license_key == "cc0-1.0":
        score += 8
    elif license_key == "cc-by-4.0":
        score += 7
    elif license_key == "cc-by-sa-4.0":
        score += 6
    elif license_key == "cc-by-3.0":
        score += 5
    elif license_key == "cc-by-sa-3.0":
        score += 4

    if preferred_country and occ.get("countryCode") == preferred_country:
        score += 4

    issues = occ.get("issues") or []
    if isinstance(issues, list) and len(issues) == 0:
        score += 3
    elif isinstance(issues, list) and len(issues) <= 2:
        score += 1

    if occ.get("identifiedBy"):
        score += 2
    if occ.get("eventDate"):
        score += 1
    if occ.get("decimalLatitude") and occ.get("decimalLongitude"):
        score += 1

    if media_obj.get("creator"):
        score += 2
    if media_obj.get("rightsHolder"):
        score += 1
    if media_obj.get("references"):
        score += 1

    fmt = (media_obj.get("format") or "").lower()
    if "jpeg" in fmt or "jpg" in fmt:
        score += 1

    return score


def slugify(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "plant"


def load_synonym_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    root = data.get("scientificNameToAccepted") if isinstance(data, dict) else None
    if not isinstance(root, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in root.items():
        ks = str(k).strip().lower()
        vs = str(v).strip()
        if ks and vs:
            out[ks] = vs
    return out


def canonical_scientific(name: str, synonym_map: Dict[str, str]) -> str:
    n = str(name or "").strip()
    if not n:
        return n
    return synonym_map.get(n.lower(), n)


def guess_extension(media_obj: Dict[str, Any], identifier: str) -> str:
    fmt = (media_obj.get("format") or "").lower()
    if "jpeg" in fmt or "jpg" in fmt:
        return ".jpg"
    if "png" in fmt:
        return ".png"
    if "webp" in fmt:
        return ".webp"
    if "tiff" in fmt or "tif" in fmt:
        return ".tif"
    if "gif" in fmt:
        return ".gif"

    m = re.search(r"\.(jpg|jpeg|png|webp|tif|tiff|gif)(\?|$)", identifier, re.IGNORECASE)
    if m:
        ext = m.group(1).lower()
        if ext == "jpeg":
            ext = "jpg"
        if ext == "tiff":
            ext = "tif"
        return f".{ext}"
    return ".jpg"


def iter_plants(resolved_path: Path) -> Iterable[Dict[str, Any]]:
    data = json.loads(resolved_path.read_text(encoding="utf-8"))
    for row in data:
        if row.get("taxonKey"):
            yield row


def fetch_page(
    session: requests.Session,
    taxon_key: int,
    offset: int,
    limit: int,
    restrict_country: Optional[str],
    year_from: Optional[int],
    year_to: Optional[int],
    timeout: int,
) -> Dict[str, Any]:
    params = {
        "taxonKey": taxon_key,
        "limit": limit,
        "offset": offset,
        "mediaType": MEDIA_TYPE,
    }
    if restrict_country:
        params["country"] = restrict_country
    if year_from and year_to:
        params["year"] = f"{year_from},{year_to}"
    return http_get(session, "/occurrence/search", params=params, timeout=timeout)


def select_candidates(
    session: requests.Session,
    plant: Dict[str, Any],
    limit: int,
    max_pages: int,
    min_candidates: int,
    restrict_country: Optional[str],
    preferred_country: Optional[str],
    year_from: Optional[int],
    year_to: Optional[int],
    sleep_seconds: float,
    timeout: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    taxon_key = plant["taxonKey"]

    for page in range(max_pages):
        offset = page * limit
        data = fetch_page(
            session,
            taxon_key,
            offset,
            limit,
            restrict_country,
            year_from,
            year_to,
            timeout,
        )
        results = data.get("results", [])
        if not results:
            break

        for occ in results:
            occ_year = occ.get("year")
            if year_from and year_to and occ_year:
                if occ_year < year_from or occ_year > year_to:
                    continue
            occ_key = occ.get("key")
            media_list = occ.get("media") or []
            if not isinstance(media_list, list) or not media_list:
                continue

            for m in media_list:
                if not isinstance(m, dict):
                    continue

                ok, lic_key, lic_label = media_is_commercial_ok(m)
                if not ok:
                    continue

                identifier = (m.get("identifier") or "").strip()
                if not identifier:
                    continue

                dedup_key = (str(occ_key), identifier)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                rows.append(
                    {
                        "quality_score": quality_score(occ, m, lic_key, preferred_country),
                        "plant_scientific": plant.get("scientificName"),
                        "plant_de": plant.get("de"),
                        "taxonKey": taxon_key,

                        "occurrenceKey": occ_key,
                        "scientificName": occ.get("scientificName"),
                        "datasetKey": occ.get("datasetKey"),
                        "publishingOrgKey": occ.get("publishingOrgKey"),
                        "publisher": occ.get("publisher"),
                        "basisOfRecord": occ.get("basisOfRecord"),
                        "eventDate": occ.get("eventDate"),
                        "year": occ.get("year"),
                        "month": occ.get("month"),
                        "day": occ.get("day"),
                        "countryCode": occ.get("countryCode"),
                        "stateProvince": occ.get("stateProvince"),
                        "locality": occ.get("locality"),
                        "decimalLatitude": occ.get("decimalLatitude"),
                        "decimalLongitude": occ.get("decimalLongitude"),
                        "coordinateUncertaintyInMeters": occ.get("coordinateUncertaintyInMeters"),
                        "identifiedBy": occ.get("identifiedBy"),
                        "recordedBy": occ.get("recordedBy"),
                        "issues": safe_json(occ.get("issues") or []),

                        "media_identifier": identifier,
                        "media_type": m.get("type"),
                        "media_format": m.get("format"),
                        "media_license_url": m.get("license"),
                        "media_license_key": lic_key,
                        "media_license_label": lic_label,
                        "media_creator": m.get("creator"),
                        "media_rightsHolder": m.get("rightsHolder"),
                        "media_title": m.get("title"),
                        "media_description": m.get("description"),
                        "media_references": m.get("references"),

                        "attribution_suggestion": guess_attribution(occ, m, lic_label),
                        "year_range_used": f"{year_from}-{year_to}" if year_from and year_to else "",
                    }
                )

        if data.get("endOfRecords") is True:
            break
        if len(rows) >= min_candidates:
            break
        time.sleep(sleep_seconds)

    return rows


def build_filename(row: Dict[str, Any]) -> str:
    plant = slugify(row.get("plant_scientific_canonical") or row.get("plant_scientific") or "plant")
    occ_key = row.get("occurrenceKey") or "occ"
    identifier = row.get("media_identifier") or ""
    lic_key = row.get("media_license_key") or "lic"
    short_hash = hashlib.sha1(identifier.encode("utf-8")).hexdigest()[:16]
    ext = guess_extension({"format": row.get("media_format")}, identifier)
    return f"{plant}__{occ_key}__{short_hash}__{lic_key}{ext}"


def download_image(
    session: requests.Session,
    row: Dict[str, Any],
    out_dir: Path,
    timeout: int,
    retries: int,
    retry_sleep: float,
) -> Optional[Path]:
    identifier = row.get("media_identifier") or ""
    if not identifier:
        return None
    plant_dir = out_dir / slugify(row.get("plant_scientific_canonical") or row.get("plant_scientific") or "plant")
    plant_dir.mkdir(parents=True, exist_ok=True)
    filename = build_filename(row)
    out_path = plant_dir / filename
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    headers = {"User-Agent": UA}
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            r = session.get(identifier, headers=headers, timeout=timeout, stream=True)
            r.raise_for_status()
            with out_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 128):
                    if chunk:
                        f.write(chunk)
            return out_path
        except requests.RequestException as exc:
            last_err = exc
            if out_path.exists():
                try:
                    out_path.unlink()
                except OSError:
                    pass
            time.sleep(retry_sleep * (attempt + 1))
            continue
    if last_err:
        print(f"Download failed after {retries} attempts: {identifier} | {last_err}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GBIF image samples for all plants.")
    parser.add_argument(
        "--resolved",
        default="gbif_wf_data/data/plants_resolved_edible.json",
        help="Path to plants_resolved_edible.json",
    )
    parser.add_argument(
        "--config",
        default="gbif_wf_data/data/gbif_download_config.json",
        help="Path to gbif_download_config.json",
    )
    parser.add_argument("--limit-per-plant", type=int, default=100)
    parser.add_argument("--min-candidates", type=int, default=80)
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--sleep-search", type=float, default=DEFAULT_SLEEP_SEARCH)
    parser.add_argument("--sleep-download", type=float, default=DEFAULT_SLEEP_DOWNLOAD)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--batch-sleep", type=float, default=DEFAULT_BATCH_SLEEP)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--download-retries", type=int, default=DEFAULT_DOWNLOAD_RETRIES)
    parser.add_argument("--download-retry-sleep", type=float, default=DEFAULT_DOWNLOAD_RETRY_SLEEP)
    parser.add_argument("--years-back", type=int, default=3)
    parser.add_argument("--min-for-random", type=int, default=300)
    parser.add_argument("--year-from", type=int, default=0)
    parser.add_argument("--year-to", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--restrict-country",
        default="",
        help="If set, restrict GBIF search to this country code (e.g., DE).",
    )
    parser.add_argument(
        "--preferred-country",
        default="",
        help="If set, boost scores for occurrences in this country.",
    )
    parser.add_argument(
        "--out-dir",
        default="gbif_wf_data/assets/plant_images/gbif_samples",
        help="Output directory for downloaded images (plant subfolders).",
    )
    parser.add_argument(
        "--meta-json",
        default="gbif_wf_data/assets/plant_images/gbif_samples/index.json",
        help="Output metadata JSON.",
    )
    parser.add_argument(
        "--meta-csv",
        default="gbif_wf_data/assets/plant_images/gbif_samples/index.csv",
        help="Output metadata CSV.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Only build metadata and skip image downloads.",
    )
    parser.add_argument(
        "--max-plants",
        type=int,
        default=0,
        help="If set, limit processing to first N plants (for testing).",
    )
    parser.add_argument(
        "--synonyms-json",
        default=str(SYNONYMS_JSON_DEFAULT),
        help="Path to centralized synonyms.json (main app assets).",
    )

    args = parser.parse_args()

    resolved_path = Path(args.resolved)
    config_path = Path(args.config)
    out_dir = Path(args.out_dir)
    meta_json = Path(args.meta_json)
    meta_csv = Path(args.meta_csv)
    synonyms_map = load_synonym_map(Path(args.synonyms_json))

    if not resolved_path.exists():
        raise SystemExit(f"Missing resolved file: {resolved_path}")

    preferred_country = args.preferred_country or ""
    restrict_country = args.restrict_country or ""

    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        if not preferred_country:
            preferred_country = cfg.get("country", "") or ""

    session = requests.Session()
    all_rows: List[Dict[str, Any]] = []
    downloads: List[Dict[str, Any]] = []
    rng = random.Random(args.seed or None)

    current_year = time.gmtime().tm_year
    year_from = args.year_from or (current_year - args.years_back + 1)
    year_to = args.year_to or current_year
    extended_year_from = args.year_from or (current_year - (args.years_back * 2) + 1)
    extended_year_to = args.year_to or current_year

    plants = list(iter_plants(resolved_path))
    if args.max_plants and args.max_plants > 0:
        plants = plants[: args.max_plants]

    seen_plant_targets = set()
    for idx, plant in enumerate(plants, start=1):
        plant_name = plant.get("scientificName")
        plant_name_canonical = canonical_scientific(plant_name or "", synonyms_map)
        print(f"[{idx}/{len(plants)}] {plant_name}")
        plant_slug = slugify(plant_name_canonical or "plant")
        dedup_key = (plant.get("taxonKey"), plant_slug)
        if dedup_key in seen_plant_targets:
            print(f"  Skipping duplicate synonym target -> {plant_name_canonical}")
            continue
        seen_plant_targets.add(dedup_key)
        plant_dir = out_dir / plant_slug
        selected_dir = plant_dir / "selected"
        selected_light_dir = plant_dir / "selected_light"
        selected_dir.mkdir(parents=True, exist_ok=True)
        selected_light_dir.mkdir(parents=True, exist_ok=True)

        existing_images = [
            p
            for p in plant_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".gif"}
        ]
        if len(existing_images) >= args.limit_per_plant:
            print(
                f"  Skipping download (already {len(existing_images)} images in {plant_dir})."
            )
            continue
        remaining = max(args.limit_per_plant - len(existing_images), 0)
        candidates = select_candidates(
            session=session,
            plant=plant,
            limit=args.limit,
            max_pages=args.max_pages,
            min_candidates=args.min_candidates,
            restrict_country=restrict_country or None,
            preferred_country=preferred_country or None,
            year_from=year_from,
            year_to=year_to,
            sleep_seconds=args.sleep_search,
            timeout=args.timeout,
        )
        if len(candidates) < args.min_for_random:
            candidates = select_candidates(
                session=session,
                plant=plant,
                limit=args.limit,
                max_pages=args.max_pages,
                min_candidates=args.min_candidates,
                restrict_country=restrict_country or None,
                preferred_country=preferred_country or None,
                year_from=extended_year_from,
                year_to=extended_year_to,
                sleep_seconds=args.sleep_search,
                timeout=args.timeout,
            )
        rng.shuffle(candidates)
        rng.shuffle(candidates)
        selected = candidates[:remaining]
        for row in selected:
            row["selected"] = True
            row["plant_scientific_canonical"] = plant_name_canonical
        for row in candidates[remaining:]:
            row["selected"] = False
            row["plant_scientific_canonical"] = plant_name_canonical
        all_rows.extend(candidates)

        if args.no_download:
            continue

        for i, row in enumerate(selected, start=1):
            out_path = download_image(
                session,
                row,
                out_dir,
                args.timeout,
                retries=args.download_retries,
                retry_sleep=args.download_retry_sleep,
            )
            row["local_path"] = str(out_path) if out_path else ""
            downloads.append(row)
            time.sleep(args.sleep_download)
            if args.batch_size > 0 and i % args.batch_size == 0:
                time.sleep(args.batch_sleep)

    meta_json.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "plants_count": len(plants),
        "limit_per_plant": args.limit_per_plant,
        "min_candidates": args.min_candidates,
        "max_pages": args.max_pages,
        "limit": args.limit,
        "year_from": year_from,
        "year_to": year_to,
        "years_back": args.years_back,
        "seed": args.seed,
        "preferred_country": preferred_country,
        "restrict_country": restrict_country,
        "out_dir": str(out_dir),
        "downloads": downloads,
    }
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    write_csv(all_rows, meta_csv)

    print("Done.")
    print(f"Metadata JSON: {meta_json}")
    print(f"Metadata CSV: {meta_csv}")
    if not args.no_download:
        print(f"Images dir: {out_dir}")


if __name__ == "__main__":
    main()
