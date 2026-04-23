# Download Missing-Light Images (Current)

This repository no longer uses `tools/download_missing_plants_images.py`.
Use the active pair:

- `tools/list_missing_light_plants.py`
- `tools/download_missing_light_images.py`

## Step 1: Generate missing list

```bash
cd gbif_wf_data
python3 tools/list_missing_light_plants.py \
  --curated-root assets/plant_images/light_curated \
  --out data/missing_light_plants.json
```

## Step 2: Dry run download scope

```bash
cd gbif_wf_data
python3 tools/download_missing_light_images.py \
  --missing-list data/missing_light_plants.json \
  --resolved data/plants_resolved_edible.json \
  --out-root assets/plant_images/light_candidates \
  --dry-run
```

## Step 3: Download images

```bash
cd gbif_wf_data
python3 tools/download_missing_light_images.py \
  --missing-list data/missing_light_plants.json \
  --resolved data/plants_resolved_edible.json \
  --out-root assets/plant_images/light_candidates
```

## Common options

```bash
# More candidates per species
--limit 2000

# Max selected images per plant
--limit-per-plant 30

# Restrict to one country
--restrict-country DE

# Prefer one country while still allowing fallback
--preferred-country DE
```

## Next step

After downloads complete, build light release zips with:

```bash
python3 tools/build_light_zips_from_curated.py --help
```
