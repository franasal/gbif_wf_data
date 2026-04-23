# Legacy tools

This folder now keeps only one script that is still used by CI:

- `publish_image_license_inventory.py`
  - Used by `.github/workflows/legal-inventory.yml`
  - Builds legal inventory artifacts from image manifests
  - Optional Firestore publish for legal metadata

All former duplicated/obsolete pipeline scripts were removed. The active GBIF
pipeline scripts live directly under `tools/` and are documented in:

- `gbif_wf_data/tools/README.md`
- `gbif_wf_data/README.md`
