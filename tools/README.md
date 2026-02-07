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
- It also writes `data/legal/image_license_todo.json` with actionable entries missing license/source-page/creator fields.

## Light build size tuning

`build_light_release.py` supports two knobs to reduce release size:

- `--max-long-edge`: caps the longer image side (works for portrait + landscape).
- `--max-images-per-plant`: limits images kept per plant folder.

Example compact profile:

```bash
python tools/build_light_release.py \
  --max-long-edge 896 \
  --quality 70 \
  --max-images-per-plant 3 \
  --zip-path gbif_wf_data/assets/plant_images/light_build/gbif_light_896_q70_cap3.zip
```
