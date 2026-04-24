# GBIF daily pipeline (overview and updates)

This repository hosts a daily GBIF download + export pipeline that:

1. Uses curated, manually maintained GBIF taxon resolution files.
2. Downloads an incremental DWCA export from GBIF using LAST_INTERPRETED filters.
3. Loads the DWCA `occurrence.txt` into SQLite.
4. Exports a compact JSON dataset (sampled points per plant and per cell).
5. Produces a daily updates summary for newly interpreted points.
6. Builds the protected-areas asset from the official BfN map service.
7. Publishes the compact dataset and protected-areas asset as GitHub Release assets.

The pipeline is orchestrated by GitHub Actions and driven by the scripts in `tools/`.

For a project-level reference (including pre-sampling visualization and current downsampling settings), see:
- `../GBIF_DATA_PROCESSING.md`

---

## Pipeline flow (end-to-end)

### 1) Inputs you maintain

| File | Purpose |
| --- | --- |
| `data/names_edible.json` | Scientific name → German/common name mapping for edible plants (authoritative edible list). |
| `data/names_poisonous.json` | Scientific name → German/common name mapping for poisonous plants (authoritative poisonous list). |
| `data/gbif_download_config.json` | Country, (optional) year range, rolling window days, daily window length, weekly refresh weekday, export sampling parameters, and gzip settings. |

### 2) Curated taxon resolution (`tools/resolve_taxa.py`)

- The daily pipeline does **not** resolve taxa automatically.
- Instead it reads the curated files:
  - `data/plants_resolved_edible.json`
  - `data/plants_resolved_poisonous.json`
- `data/taxon_cache.json` remains useful as supporting metadata, especially when a resolved taxon is a synonym and the exporter/download predicate needs the accepted usage key as well.

#### Manual workflow when plant lists change

1. Edit `data/names_edible.json` or `data/names_poisonous.json`.
2. Run the resolver manually for the changed list.
3. Review the output match metadata.
4. Commit the updated `plants_resolved_*.json` and `taxon_cache.json`.
5. Re-run the normal daily/export pipeline only after the curated resolved files are in sync.

Daily pipeline behavior:
- Missing current names in `plants_resolved_*.json` are a hard error.
- Extra legacy or synonym rows in `plants_resolved_*.json` are warned about and ignored during daily runs, but should still be cleaned up during manual taxonomy maintenance.

Example commands:

```bash
python tools/resolve_taxa.py \
  --names data/names_edible.json \
  --out data/plants_resolved_edible.json \
  --cache data/taxon_cache.json

python tools/resolve_taxa.py \
  --names data/names_poisonous.json \
  --out data/plants_resolved_poisonous.json \
  --cache data/taxon_cache.json
```

**Output shape (example):**

```json
[
  {
    "scientificName": "Acer platanoides",
    "de": "Spitz-Ahorn",
    "taxonKey": 2877951,
    "match": {
      "matchType": "EXACT",
      "confidence": 98,
      "canonicalName": "Acer platanoides",
      "scientificName": "Acer platanoides",
      "rank": "SPECIES",
      "status": "ACCEPTED"
    }
  }
]
```

### 3) GBIF download request + polling (`tools/run_daily_update.py`)

- Builds a GBIF predicate that includes:
  - `HAS_COORDINATE = true` (if configured)
  - Country + year range filters
  - `TAXON_KEY in [...]` from the curated resolved plant lists
  - configured window key `>= <since>` using a daily window or a weekly full-range refresh
- Requests a DWCA download, waits for completion, and downloads the zip into `.tmp_gbif/<key>`.

### 4) SQLite load (`tools/dwca_sqlite.py`)

- Finds `occurrence.txt` in the DWCA bundle.
- Loads it into `data/dwca.sqlite` with the `occ` table and supporting indexes.
- Stores the raw TSV only if needed (pipeline uses `--no-raw` for performance).
- Optionally writes `data/changes_summary.json`, reporting how many existing records changed and which fields changed.

### 4b) Rolling window prune (optional)

- If `rolling_window_days` is set, the pipeline can delete rows older than the cutoff date after each load.
- Pruning is date-based (eventDate or year/month/day) and can be skipped with `--no-prune`.

### 5) Daily updates summary (`data/updates_summary_edible.json`, `data/updates_summary_poisonous.json`)

- Immediately after extraction, the pipeline scans the DWCA `occurrence.txt`.
- Counts how many points were **newly interpreted in the current window** (daily by default, full-range on weekly refresh), filtered to the whitelist.
- Outputs a compact JSON summary:

```json
{
  "generated_at": "2026-01-26T13:33:33Z",
  "download_key": "<gbif-download-key>",
  "interpreted_since": "2026-01-24",
  "window_start": "2026-01-25",
  "window_days": 1,
  "window_label": "daily",
  "total_new_points": 1234,
  "per_species": {
    "Acer platanoides": 42,
    "Betula pendula": 17
  }
}
```

### 6) Compact export (`tools/export_occurrences_compact.py`)

- Uses the whitelist as **source of truth** (includes all configured plants).
- Reserves the true newest observations per plant (currently 10) so latest records are guaranteed in compact `points`.
- Samples additional newest points per geohash cell per plant to fill remaining cap budget.
- Outputs `data/occurrences_compact_edible.json.gz` and `data/occurrences_compact_poisonous.json.gz` with:
  - `region` (name + center)
  - `plants` (per-plant stats + sampled points)
  - `meta` (export parameters)

**Output shape (example):**

```json
{
  "region": {
    "name": "Germany",
    "center": {"lat": 51.0, "lon": 10.0}
  },
  "plants": {
    "Acer platanoides": {
      "de": "Spitz-Ahorn",
      "taxonKey": 2877951,
      "total": 12345,
      "year_counts": {"2023": 120, "2024": 98},
      "month_counts_all": [1,2,3,4,5,6,7,8,9,10,11,12],
      "last_obs": {"year": 2024, "month": 7},
      "bbox": [48.1, 53.9, 6.2, 13.9],
      "points": [[52.5, 13.4, 2024, 7]],
      "sampled_total": 1
    }
  },
  "meta": {
    "generated_at": "2026-01-26T13:33:33Z",
    "source": "dwca.sqlite",
    "country": "DE",
    "year_from": 2023,
    "year_to": 2026,
    "top_n": 250,
    "cell_precision": 5,
    "keep_per_cell": 6,
    "max_points_per_plant": 700,
    "scanned_rows": 1234567
  }
}
```

### 7) Release publishing (`tools/publish_release_asset.py`)

- Publishes `data/occurrences_compact.json.gz` (legacy/prod) plus `data/occurrences_compact_edible.json.gz` and `data/occurrences_compact_poisonous.json.gz` (dev split) as Release assets on tag `latest`.
- Publishes `data/protected_areas_de.json.gz` as a separate legal-boundary asset on both `latest` and `latest-dev`.
- Default tags are:
  - `latest`
  - `latest-dev`
- On manual workflow runs, `release_suffix` can be provided to publish isolated test channels without touching prod/dev tags:
  - example suffix `-canary` -> tags `latest-canary` and `latest-dev-canary`

### 7b) Browser-safe asset mirror (Cloudflare R2)

- The admin panel runs in the browser and should not fetch GitHub release downloads directly because release redirects are unreliable for web CORS.
- The workflow therefore mirrors selected browser-facing JSON assets to the public R2 base already used by the app.
- Current mirrored pipeline-analysis assets:
  - `pipeline_diagnostics_summary.json`
  - `pipeline_point_explorer.json`
- Mirror helper:
  - `tools/mirror_assets_to_r2.py`
- Required workflow secrets:
  - `R2_ACCOUNT_ID`
  - `R2_ACCESS_KEY_ID`
  - `R2_SECRET_ACCESS_KEY`
  - `R2_BUCKET`
- Important:
  - `R2_ACCOUNT_ID` must be the Cloudflare storage account id from the private
    endpoint host `<account-id>.r2.cloudflarestorage.com`.
  - Do not use the public `pub-...r2.dev` token there; that is only the public
    delivery hostname, not the S3 API account id.

---

## GitHub Actions workflow summary

The workflow runs in two jobs:

1. **build_db** (db-only):
   - Verifies that `names_*.json` and `plants_resolved_*.json` are in sync.
   - Requests + downloads DWCA using a daily window (or weekly full-range refresh window).
   - Generates `updates_summary_edible.json` and `updates_summary_poisonous.json`.
   - Loads `dwca.sqlite` and uploads it as an artifact.
   - Commits state/output files back to the repo.

2. **export_and_release** (export-only):
   - Downloads the DB artifact.
   - Runs the exporter to create `occurrences_compact_edible.json.gz` and `occurrences_compact_poisonous.json.gz`.
   - Builds `protected_areas_de.json.gz` from the BfN service.
   - Publishes the Release assets.

---

## Updates performed in this cleanup/optimization pass

### ✅ Output-preserving optimizations

- **Whitelist-as-source-of-truth:**
  - The export script now includes all names from the whitelist when a `names_*.json` is provided (no `top_n` slicing).

- **Batch taxonKey lookup:**
  - Exporter now fetches taxon keys with one grouped query instead of per-species queries.

- **SQLite index enhancements:**
  - Added single-column indexes on `scientificName` and `species` to speed filtering and lookup.

- **SQLite/state checkpoint reuse in CI:**
  - The workflow persists `data/dwca.sqlite` + `data/gbif_state.json` between runs.
  - Daily runs append to the rolling DB; weekly/full refreshes can still rebuild from scratch.

### ✅ Rolling daily + weekly full-range refresh

- Daily runs download the last 24 hours of interpretations.
- Weekly runs (Wednesday 04:00 CET) refresh from `year_from` through today to capture updates to older records.
- The SQLite DB is cached between runs so daily DWCA deltas can be appended to the existing dataset.

### ✅ New daily updates summary

- The pipeline now writes `data/updates_summary_edible.json` and `data/updates_summary_poisonous.json`, which report newly interpreted points in the current download window for each list.

### ✅ Removed unused/unwired scripts

- Removed obsolete duplicated legacy pipeline scripts from `tools/legacy/`.
- Kept only `tools/legacy/publish_image_license_inventory.py` for the legal-inventory workflow.

---

## Where to look for outputs

| File | Description |
| --- | --- |
| `data/plants_resolved.json` | Name resolution results for legacy/prod combined list. |
| `data/plants_resolved_edible.json` | Name resolution results for edible plants. |
| `data/plants_resolved_poisonous.json` | Name resolution results for poisonous plants. |
| `data/taxon_cache.json` | Cached/manual GBIF match metadata used to keep synonym handling reproducible. |
| `data/dwca.sqlite` | SQLite database of occurrences. |
| `data/occurrences_compact.json.gz` | Compact export for legacy/prod (combined). |
| `data/occurrences_compact_edible.json.gz` | Compact export for edible plants. |
| `data/occurrences_compact_poisonous.json.gz` | Compact export for poisonous plants. |
| `data/protected_areas_de.json.gz` | Published German protected-areas polygons for app-side legal filtering. |
| `data/updates_summary.json` | Daily counts of newly interpreted points (legacy/prod combined). |
| `data/updates_summary_edible.json` | Daily counts of newly interpreted edible points. |
| `data/updates_summary_poisonous.json` | Daily counts of newly interpreted poisonous points. |
| `data/changes_summary.json` | Field-level change summary vs. existing DB (if enabled). |

---

## Notes on compatibility

All optimizations were implemented to **avoid changing data format or quality**. They affect **performance and repeatability** only, while keeping the output JSON schemas unchanged.

## Protected Areas Asset

The protected-areas asset is intentionally separate from the GBIF occurrence exports.

- Source: BfN "Schutzgebiete in Deutschland" ArcGIS service
- Included categories:
  - `Naturschutzgebiet`
  - `Nationalpark`
  - `Nationales Naturmonument`
- Output: `data/protected_areas_de.json.gz`
- Builder: `tools/fetch_protected_areas.py`
- Schedule: daily in the same workflow as the compact export

This separation keeps legal-boundary updates independent from biological-occurrence exports and allows the app to cache them on a separate 24-hour policy.
