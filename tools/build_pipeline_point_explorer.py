#!/usr/bin/env python3
import argparse
import gzip
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _load_json(path: Path, default: Any) -> Any:
    if not path or not path.exists():
        return default
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _normalize_doc_id(scientific_name: str, taxon_key: Any) -> str:
    tk = _safe_int(taxon_key)
    if tk:
        return f"pipeline_point_explorer_taxon_{tk}"
    slug = re.sub(r"[^a-z0-9]+", "_", scientific_name.lower()).strip("_")
    return f"pipeline_point_explorer_{slug}"


def _point_key(point: list[Any]) -> str:
    lat = point[0] if len(point) > 0 else None
    lon = point[1] if len(point) > 1 else None
    year = point[2] if len(point) > 2 else None
    month = point[3] if len(point) > 3 else None
    gbif_id = point[4] if len(point) > 4 else None
    ref = point[9] if len(point) > 9 else None
    return "|".join(str(v) for v in [lat, lon, year, month, gbif_id, ref])


def _point_to_obj(point: list[Any]) -> dict[str, Any]:
    return {
        "lat": point[0] if len(point) > 0 else None,
        "lon": point[1] if len(point) > 1 else None,
        "year": point[2] if len(point) > 2 else None,
        "month": point[3] if len(point) > 3 else None,
        "gbifId": point[4] if len(point) > 4 else None,
        "sourceLabel": point[5] if len(point) > 5 else None,
        "datasetName": point[6] if len(point) > 6 else None,
        "mediaImageUrl": point[7] if len(point) > 7 else None,
        "mediaSourcePageUrl": point[8] if len(point) > 8 else None,
        "occurrenceReferenceUrl": point[9] if len(point) > 9 else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build per-plant point explorer payloads for admin diagnostics.",
    )
    ap.add_argument("--compact", required=True)
    ap.add_argument("--diagnostics-summary", required=True)
    ap.add_argument("--loss-report", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--recent", default="")
    args = ap.parse_args()

    compact = _load_json(Path(args.compact), {})
    diagnostics_summary = _load_json(Path(args.diagnostics_summary), {})
    loss_report = _load_json(Path(args.loss_report), {})
    recent = _load_json(Path(args.recent), {}) if args.recent else {}

    compact_plants = compact.get("plants") if isinstance(compact, dict) else None
    if not isinstance(compact_plants, dict):
        compact_plants = {}
    recent_plants = recent.get("plants") if isinstance(recent, dict) else None
    if not isinstance(recent_plants, dict):
        recent_plants = {}

    diagnostics_by_sci = {}
    for row in diagnostics_summary.get("plants", []):
        if isinstance(row, dict) and row.get("scientificName"):
            diagnostics_by_sci[row["scientificName"]] = row
    loss_by_sci = {}
    for row in loss_report.get("plants", []):
        if isinstance(row, dict) and row.get("scientificName"):
            loss_by_sci[row["scientificName"]] = row

    index_plants: list[dict[str, Any]] = []
    detail_docs: dict[str, dict[str, Any]] = {}

    for scientific_name, compact_payload in compact_plants.items():
        if not isinstance(compact_payload, dict):
            continue
        taxon_key = compact_payload.get("taxonKey")
        doc_id = _normalize_doc_id(scientific_name, taxon_key)
        diag = diagnostics_by_sci.get(scientific_name, {})
        loss_row = loss_by_sci.get(scientific_name, {})
        compact_points = compact_payload.get("points")
        compact_points = compact_points if isinstance(compact_points, list) else []
        recent_payload = recent_plants.get(scientific_name, {})
        recent_points = recent_payload.get("points")
        recent_points = recent_points if isinstance(recent_points, list) else []

        recent_keys = {_point_key(p) for p in recent_points if isinstance(p, list)}
        shipped_points = []
        for point in compact_points:
            if not isinstance(point, list):
                continue
            item = _point_to_obj(point)
            item["presentInRecentRaw"] = _point_key(point) in recent_keys
            shipped_points.append(item)

        compact_keys = {_point_key(p) for p in compact_points if isinstance(p, list)}
        recent_missing_points = []
        for point in recent_points:
            if not isinstance(point, list):
                continue
            if _point_key(point) in compact_keys:
                continue
            recent_missing_points.append(_point_to_obj(point))

        detail_payload = {
            "generatedAt": _utc_now_iso(),
            "docId": doc_id,
            "scientificName": scientific_name,
            "commonName": compact_payload.get("de") or "",
            "taxonKey": taxon_key,
            "rawTotal": _safe_int(diag.get("rawTotal", compact_payload.get("total"))),
            "sampledTotal": _safe_int(
                diag.get("sampledTotal", compact_payload.get("sampled_total"))
            ),
            "droppedBySampling": _safe_int(diag.get("droppedBySampling")),
            "samplingCoverage": diag.get("samplingCoverage"),
            "sampling": compact_payload.get("sampling") or {},
            "observationSources": compact_payload.get("observation_sources") or {},
            "pointsSchema": (compact.get("meta") or {}).get("points_schema") or [],
            "shippedPoints": shipped_points,
            "recentMissingPoints": recent_missing_points,
            "recentRawAvailable": bool(recent_points),
            "recentRawPointCount": len(recent_points),
            "dropReasonCounts": loss_row.get("dropReasonCounts") or {},
            "droppedPointsSample": loss_row.get("droppedPointsSample") or [],
        }
        detail_docs[doc_id] = detail_payload
        index_plants.append(
            {
                "docId": doc_id,
                "scientificName": scientific_name,
                "commonName": compact_payload.get("de") or "",
                "taxonKey": taxon_key,
                "rawTotal": detail_payload["rawTotal"],
                "sampledTotal": detail_payload["sampledTotal"],
                "droppedBySampling": detail_payload["droppedBySampling"],
                "samplingCoverage": detail_payload["samplingCoverage"],
                "recentRawAvailable": detail_payload["recentRawAvailable"],
                "recentRawPointCount": detail_payload["recentRawPointCount"],
            }
        )

    index_plants.sort(
        key=lambda row: (
            -(row.get("droppedBySampling") or 0),
            row.get("scientificName") or "",
        )
    )

    out = {
        "generatedAt": _utc_now_iso(),
        "plantCount": len(index_plants),
        "index": index_plants,
        "details": detail_docs,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[ok] wrote {out_path} (plants={len(index_plants)})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
