#!/usr/bin/env python3
"""Fetch and build the published protected-areas asset from the BfN service."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError


REFERER = "https://geodienste.bfn.de/schutzgebiete?l=~schgeb%28-4%29&lang=de"
USER_AGENT = "Mozilla/5.0 (compatible; WildForagerProtectedAreaImporter/1.0)"
BASE_URL = (
    "https://geodienste.bfn.de/server/rest/services/bfn_sch/Schutzgebiet/MapServer"
)
REQUEST_RETRIES = 4
RETRY_DELAY_SECONDS = 2.0


@dataclass(frozen=True)
class LayerSpec:
    layer_id: int
    category: str
    legal_basis: str
    category_code: str


LAYER_SPECS = (
    LayerSpec(7, "Naturschutzgebiet", "§23 BNatSchG", "nsg"),
    LayerSpec(5, "Nationalpark", "§24 Abs. 3 BNatSchG", "np"),
    LayerSpec(0, "Nationales Naturmonument", "§24 Abs. 4 BNatSchG", "nnm"),
)


def _preview_body(raw: bytes, limit: int = 160) -> str:
    text = raw.decode("utf-8", errors="replace").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _request_json(url: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Referer": REFERER,
            "Accept": "application/json, application/geo+json",
        },
    )
    last_error: Exception | None = None
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
                content_type = resp.headers.get("Content-Type", "")
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < REQUEST_RETRIES:
                print(
                    f"Request attempt {attempt}/{REQUEST_RETRIES} failed for {url}: {exc}. "
                    f"Retrying in {RETRY_DELAY_SECONDS:.1f}s.",
                    file=sys.stderr,
                )
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            break

        if not raw.strip():
            last_error = ValueError(
                f"Empty response body (content-type={content_type or 'unknown'})",
            )
        else:
            try:
                return json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                last_error = ValueError(
                    "Non-JSON response from BfN service "
                    f"(content-type={content_type or 'unknown'}, body={_preview_body(raw)!r})",
                )

        if attempt < REQUEST_RETRIES:
            print(
                f"Request attempt {attempt}/{REQUEST_RETRIES} returned invalid JSON for {url}: "
                f"{last_error}. Retrying in {RETRY_DELAY_SECONDS:.1f}s.",
                file=sys.stderr,
            )
            time.sleep(RETRY_DELAY_SECONDS)

    raise RuntimeError(f"Failed to fetch JSON from {url}: {last_error}")


def _query_url(layer_id: int, **params: object) -> str:
    encoded = urllib.parse.urlencode(
        {key: str(value) for key, value in params.items()},
    )
    return f"{BASE_URL}/{layer_id}/query?{encoded}"


def _fetch_layer(spec: LayerSpec, out_dir: Path, batch_size: int) -> list[Path]:
    count_payload = _request_json(
        _query_url(spec.layer_id, where="1=1", returnCountOnly="true", f="json"),
    )
    count = int(count_payload["count"])
    print(f"Layer {spec.layer_id} ({spec.category_code}): {count}", file=sys.stderr)
    written: list[Path] = []
    for offset in range(0, count, batch_size):
        out_path = out_dir / f"{spec.category_code}_{offset:05d}.geojson"
        payload = _request_json(
            _query_url(
                spec.layer_id,
                where="1=1",
                returnGeometry="true",
                outFields="OBJECTID,NAME,BL,CDDA_CODE,IUCN_KAT,STATUS,LEG_DATE,JAHR",
                orderByFields="OBJECTID",
                resultOffset=offset,
                resultRecordCount=batch_size,
                geometryPrecision=5,
                maxAllowableOffset=0.0002,
                outSR=4326,
                f="geojson",
            ),
        )
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        written.append(out_path)
        print(f"  fetched offset {offset} -> {out_path.name}", file=sys.stderr)
    return written


def _normalize_feature(
    feature: dict,
    layer: LayerSpec,
    *,
    fetched_at: str,
    source_version: str,
) -> dict:
    props = feature.get("properties") or {}
    geometry = feature.get("geometry")
    if not geometry:
        return {}
    object_id = props.get("OBJECTID")
    cdda_code = props.get("CDDA_CODE")
    feature_id = cdda_code if cdda_code not in (None, "") else object_id
    if feature_id in (None, ""):
        raise ValueError(f"Missing feature id for layer {layer.layer_id}")
    return {
        "type": "Feature",
        "properties": {
            "id": f"{layer.category_code}-{feature_id}",
            "name": (props.get("NAME") or "").strip(),
            "category": layer.category,
            "categoryCode": layer.category_code,
            "bundesland": props.get("BL"),
            "cddaCode": cdda_code,
            "iucnCategory": props.get("IUCN_KAT"),
            "status": props.get("STATUS"),
            "legalBasis": layer.legal_basis,
            "source": "Bundesamt fuer Naturschutz (BfN)",
            "sourceDataset": "Schutzgebiete in Deutschland / MapServer",
            "sourceLayerId": layer.layer_id,
            "sourceVersion": source_version,
            "fetchedAt": fetched_at,
            "foragingRestricted": True,
        },
        "geometry": geometry,
    }


def _spec_by_code(code: str) -> LayerSpec:
    for spec in LAYER_SPECS:
        if spec.category_code == code:
            return spec
    raise KeyError(f"Unknown layer code: {code}")


def _load_raw_features(input_dir: Path) -> list[tuple[LayerSpec, dict]]:
    collected: list[tuple[LayerSpec, dict]] = []
    for raw_path in sorted(input_dir.glob("*.geojson")):
        code = raw_path.stem.split("_", 1)[0]
        spec = _spec_by_code(code)
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        for feature in payload.get("features") or []:
            collected.append((spec, feature))
    return collected


def build_asset(output_path: Path, input_dir: Path) -> None:
    fetched_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    source_version = fetched_at.split("T", 1)[0]
    features = []

    for layer, raw_feature in _load_raw_features(input_dir):
        normalized = _normalize_feature(
            raw_feature,
            layer,
            fetched_at=fetched_at,
            source_version=source_version,
        )
        if normalized:
            features.append(normalized)

    output = {
        "type": "FeatureCollection",
        "name": "wild_forager_protected_areas_de",
        "metadata": {
            "country": "DE",
            "scope": [layer.category for layer in LAYER_SPECS],
            "fetchedAt": fetched_at,
            "sourceVersion": source_version,
            "source": "Bundesamt fuer Naturschutz (BfN)",
            "sourceUrl": "https://www.bfn.de/schutzgebiete",
            "legalScopeNote": (
                "Only categories that are protected like Naturschutzgebiete are included. "
                "Biosphaerenreservate, Naturparke, Landschaftsschutzgebiete and Natura 2000 "
                "are intentionally excluded from automatic blocking."
            ),
        },
        "features": features,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, separators=(",", ":"))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {output_path} ({size_mb:.2f} MiB)", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="data/protected_areas_de.json.gz",
        help="Output path for the published gzip asset.",
    )
    parser.add_argument(
        "--tmp-dir",
        default="tmp/protected_areas_raw",
        help="Directory used for the raw BfN layer exports.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="ArcGIS query page size.",
    )
    args = parser.parse_args()

    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for raw_path in tmp_dir.glob("*.geojson"):
        raw_path.unlink()

    for spec in LAYER_SPECS:
        _fetch_layer(spec, tmp_dir, batch_size=args.batch_size)

    build_asset(Path(args.output), tmp_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
