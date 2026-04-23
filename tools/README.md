# GBIF tools

Active scripts in this folder:

- `run_daily_update.py` - Orchestrates GBIF download, SQLite load, and compact export from curated resolved taxa files.
- `resolve_taxa.py` - Manual helper to resolve scientific names to GBIF taxon keys when the curated plant list changes.
- `dwca_sqlite.py` - Loads DWCA `occurrence.txt` into `data/dwca.sqlite`.
- `export_occurrences_compact.py` - Builds compact map dataset exports.
- `export_approved_observations_from_firestore.py` - Exports approved observation overlay.
- `fetch_protected_areas.py` - Fetches BfN protected-area layers and builds `protected_areas_de.json.gz`.
- `generate_approved_observations_placeholder.py` - Writes a placeholder overlay when export is unavailable.
- `publish_release_asset.py` - Publishes release assets.
- `visualize_gbif_download.py` - Generates pre-sampling whole-download summaries and grid GeoJSON.
- `list_missing_light_plants.py`, `download_missing_light_images.py`, `build_light_zips_from_curated.py` - Light image workflow scripts.

Primary docs:
- `gbif_wf_data/README.md`
- `docs/gbif_light_pipeline_v2.md`

Legacy scripts:
- `tools/legacy/publish_image_license_inventory.py` is retained for the legal inventory workflow.
