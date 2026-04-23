# Light Build Guide (Current)

This repository no longer ships `tools/build_light_external.py`.
That legacy helper depended on removed legacy build scripts.

Use the active light pipeline scripts instead.

## 1) Build missing-plant list

```bash
cd gbif_wf_data
python3 tools/list_missing_light_plants.py \
  --curated-root assets/plant_images/light_curated \
  --out data/missing_light_plants.json
```

## 2) Optionally fetch images for missing plants

```bash
cd gbif_wf_data
python3 tools/download_missing_light_images.py \
  --missing-list data/missing_light_plants.json \
  --resolved data/plants_resolved_edible.json \
  --out-root assets/plant_images/light_candidates
```

## 3) Build release light ZIPs

```bash
cd gbif_wf_data
python3 tools/build_light_zips_from_curated.py \
  --curated-root assets/plant_images/light_curated \
  --index-csv assets/plant_images/gbif_samples/index.csv \
  --names-edible data/names_edible.json \
  --names-poisonous data/names_poisonous.json \
  --split-by-class \
  --output-root assets/plant_images/light_build
```

## Size tuning knobs

Use these options on `build_light_zips_from_curated.py`:

- `--max-long-edge`
- `--quality`
- `--max-images-per-plant`

Example compact profile:

```bash
python3 tools/build_light_zips_from_curated.py \
  --curated-root assets/plant_images/light_curated \
  --index-csv assets/plant_images/gbif_samples/index.csv \
  --names-edible data/names_edible.json \
  --names-poisonous data/names_poisonous.json \
  --split-by-class \
  --max-long-edge 896 \
  --quality 70 \
  --max-images-per-plant 3
```
