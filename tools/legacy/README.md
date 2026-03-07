# Pipeline tools

The GitHub Actions workflow uses these scripts:

- `run_daily_update.py` to download GBIF data, update the SQLite database, and export the compact datasets (edible + poisonous).
- `export_occurrences_compact.py` for the compact JSON export (invoked by `run_daily_update.py`).
- `resolve_taxa.py` and `dwca_sqlite.py` during the database build step.
- `publish_release_asset.py` to publish the release assets (edible + poisonous).

Helper scripts that are not part of the GitHub Actions workflow are kept here when they support curation and release prep.

The DB step also writes `data/updates_summary.json` (legacy/prod combined) and `data/updates_summary_edible.json`/`data/updates_summary_poisonous.json` (split), which summarize new points interpreted in the current download window for each list (daily window or weekly full-range refresh).

Additional tooling:

- `publish_image_license_inventory.py` builds a versioned legal inventory from image manifests and can publish metadata to Firestore.
- It also writes `data/legal/image_license_todo.json` with actionable entries missing license/source-page/creator fields.
- `gbif_mark_curation_prefixes.py` renames plant folders with curation/status prefixes and updates related index/manifest metadata paths.

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
