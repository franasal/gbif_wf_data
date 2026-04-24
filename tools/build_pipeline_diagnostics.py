#!/usr/bin/env python3
import argparse
import gzip
import json
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
    if not path.exists():
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


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build compact pipeline diagnostics summary for admin panel consumption.",
    )
    ap.add_argument("--compact", required=True)
    ap.add_argument("--updates-summary", required=True)
    ap.add_argument("--changes-summary", required=True)
    ap.add_argument("--pipeline-run-summary", required=True)
    ap.add_argument("--loss-report", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    compact = _load_json(Path(args.compact), {})
    updates_summary = _load_json(Path(args.updates_summary), {})
    changes_summary = _load_json(Path(args.changes_summary), {})
    pipeline_run_summary = _load_json(Path(args.pipeline_run_summary), {})
    loss_report = _load_json(Path(args.loss_report), {})

    plants = compact.get("plants") if isinstance(compact, dict) else None
    if not isinstance(plants, dict):
        plants = {}

    per_plant: list[dict[str, Any]] = []
    total_raw = 0
    total_sampled = 0
    total_raw_cells = 0
    total_shipped_cells = 0
    loss_by_sci = {}
    for row in loss_report.get("plants", []):
        if isinstance(row, dict) and row.get("scientificName"):
            loss_by_sci[row["scientificName"]] = row

    for scientific_name, payload in plants.items():
        if not isinstance(payload, dict):
            continue
        loss_row = loss_by_sci.get(scientific_name, {})
        raw_total = _safe_int(loss_row.get("rawTotal", payload.get("total")))
        sampled_total = _safe_int(
            loss_row.get("shippedTotal", payload.get("sampled_total"))
        )
        sampling = payload.get("sampling") if isinstance(payload.get("sampling"), dict) else {}
        raw_occupied_cells = _safe_int(
            loss_row.get("rawOccupiedCells", sampling.get("raw_occupied_cells"))
        )
        shipped_occupied_cells = _safe_int(
            loss_row.get("shippedOccupiedCells", sampling.get("shipped_occupied_cells"))
        )
        dropped = max(0, raw_total - sampled_total)
        total_raw += raw_total
        total_sampled += sampled_total
        total_raw_cells += raw_occupied_cells
        total_shipped_cells += shipped_occupied_cells

        observation_sources = (
            payload.get("observation_sources")
            if isinstance(payload.get("observation_sources"), dict)
            else {}
        )
        source_scoped_samples = (
            payload.get("source_scoped_samples")
            if isinstance(payload.get("source_scoped_samples"), dict)
            else {}
        )

        per_plant.append(
            {
                "scientificName": scientific_name,
                "commonName": payload.get("de") or "",
                "taxonKey": payload.get("taxonKey"),
                "rawTotal": raw_total,
                "sampledTotal": sampled_total,
                "droppedBySampling": dropped,
                "samplingCoverage": (
                    round(sampled_total / raw_total, 6) if raw_total > 0 else None
                ),
                "rawOccupiedCells": raw_occupied_cells,
                "shippedOccupiedCells": shipped_occupied_cells,
                "droppedOccupiedCells": max(0, raw_occupied_cells - shipped_occupied_cells),
                "gridCoverage": (
                    round(shipped_occupied_cells / raw_occupied_cells, 6)
                    if raw_occupied_cells > 0
                    else None
                ),
                "gridPrecision": _safe_int(
                    loss_row.get("sampling", {}).get("grid_precision")
                    if isinstance(loss_row.get("sampling"), dict)
                    else sampling.get("grid_precision")
                ),
                "latestReserved": _safe_int(sampling.get("latest_injected_count")),
                "keepPerCell": _safe_int(sampling.get("keep_per_cell")),
                "maxPointsPerPlant": _safe_int(sampling.get("max_points_per_plant")),
                "samplingMode": sampling.get("mode"),
                "samplingBucket": sampling.get("bucket"),
                "sourceCount": len(observation_sources),
                "observationSources": observation_sources,
                "sourceScopedSampleSources": sorted(source_scoped_samples.keys()),
                "sourceScopedSampleCount": sum(
                    len(points) for points in source_scoped_samples.values()
                    if isinstance(points, list)
                ),
                "droppedByDuplicate": _safe_int(
                    (loss_row.get("dropReasonCounts") or {}).get("dropped_duplicate")
                ),
                "droppedByCellQuota": _safe_int(
                    (loss_row.get("dropReasonCounts") or {}).get("dropped_cell_quota")
                ),
                "droppedByGlobalCap": _safe_int(
                    (loss_row.get("dropReasonCounts") or {}).get("dropped_global_cap")
                ),
                "dropReasonCounts": loss_row.get("dropReasonCounts") or {},
                "droppedPointsSampleCount": len(loss_row.get("droppedPointsSample") or []),
            }
        )

    per_plant.sort(
        key=lambda row: (
            -_safe_int(row.get("droppedBySampling")),
            row.get("scientificName") or "",
        )
    )

    dropped_total = max(0, total_raw - total_sampled)
    loss_totals = loss_report.get("totals") if isinstance(loss_report.get("totals"), dict) else {}
    summary = {
        "generatedAt": _utc_now_iso(),
        "window": {
            "start": pipeline_run_summary.get("window_start"),
            "end": pipeline_run_summary.get("window_end"),
            "days": _safe_int(pipeline_run_summary.get("window_days")),
            "label": pipeline_run_summary.get("window_label"),
            "field": pipeline_run_summary.get("window_field"),
            "rollingWindowDays": _safe_int(
                pipeline_run_summary.get("rolling_window_days")
            ),
            "pruneCutoff": pipeline_run_summary.get("prune_cutoff"),
        },
        "totals": {
            "plants": len(per_plant),
            "rawPointsVisibleWindow": total_raw,
            "sampledPointsShipped": total_sampled,
            "droppedBySampling": dropped_total,
            "keptLatestReserved": _safe_int(loss_totals.get("kept_latest_reserved")),
            "keptGeohashSample": _safe_int(loss_totals.get("kept_geohash_sample")),
            "droppedByDuplicate": _safe_int(loss_totals.get("dropped_duplicate")),
            "droppedByCellQuota": _safe_int(loss_totals.get("dropped_cell_quota")),
            "droppedByGlobalCap": _safe_int(loss_totals.get("dropped_global_cap")),
            "samplingCoverage": (
                round(total_sampled / total_raw, 6) if total_raw > 0 else None
            ),
            "rawOccupiedCells": total_raw_cells,
            "shippedOccupiedCells": total_shipped_cells,
            "droppedOccupiedCells": max(0, total_raw_cells - total_shipped_cells),
            "gridCoverage": (
                round(total_shipped_cells / total_raw_cells, 6)
                if total_raw_cells > 0
                else None
            ),
            "newPointsInCurrentWindow": _safe_int(
                updates_summary.get("total_new_points")
            ),
            "rowsNewInDbLoad": _safe_int(changes_summary.get("rows_new")),
            "rowsUpdatedInDbLoad": _safe_int(changes_summary.get("rows_updated")),
            "rowsScannedInDbLoad": _safe_int(changes_summary.get("rows_scanned")),
            "rowsPrunedByRollingWindow": _safe_int(
                pipeline_run_summary.get("pruned_rows")
            ),
            "dbRowsBeforePrune": _safe_int(
                pipeline_run_summary.get("db_rows_before_prune")
            ),
            "dbRowsAfterPrune": _safe_int(
                pipeline_run_summary.get("db_rows_after_prune")
            ),
        },
        "dropReasons": {
            "dropped_duplicate": _safe_int(loss_totals.get("dropped_duplicate")),
            "dropped_cell_quota": _safe_int(loss_totals.get("dropped_cell_quota")),
            "dropped_global_cap": _safe_int(loss_totals.get("dropped_global_cap")),
        },
        "changes": {
            "fieldsChanged": changes_summary.get("fields_changed", {}),
        },
        "topPlantsBySamplingLoss": per_plant[:25],
        "plants": per_plant,
        "sources": {
            "compactGeneratedAt": (compact.get("meta") or {}).get("generated_at"),
            "updatesGeneratedAt": updates_summary.get("generated_at"),
            "changesGeneratedAt": changes_summary.get("generated_at"),
            "pipelineRunGeneratedAt": pipeline_run_summary.get("generated_at"),
            "lossReportGeneratedAt": loss_report.get("generatedAt"),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[ok] wrote {out_path} (plants={len(per_plant)})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
