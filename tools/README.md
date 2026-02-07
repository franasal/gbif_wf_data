# Pipeline tools

The GitHub Actions workflow uses these scripts:

- `run_daily_update.py` to download GBIF data, update the SQLite database, and export the compact dataset.
- `export_occurrences_compact.py` for the compact JSON export (invoked by `run_daily_update.py`).
- `resolve_taxa.py` and `dwca_sqlite.py` during the database build step.
- `publish_release_asset.py` to publish the release asset.

Legacy helper scripts that are not part of the GitHub Actions workflow have been removed to keep this directory focused.

The DB step also writes `data/updates_summary.json`, which summarizes new points interpreted in the current download window for the configured plant list (daily window or weekly full-range refresh).

Additional tooling:

- `publish_image_license_inventory.py` builds a versioned legal inventory from image manifests and can publish metadata to Firestore.
