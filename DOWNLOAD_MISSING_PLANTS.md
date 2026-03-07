# Downloading GBIF Images for Missing Plants

## Quick Start

I've created a new helper script that makes it easy to download images for all the "missing_" plants.

### Step 1: Test the Script (Dry Run)

First, see what plants will be processed without actually downloading anything:

```bash
cd gbif_wf_data
python3 tools/download_missing_plants_images.py --dry-run
```

**Expected output:**
- List of all 23 missing plants found
- Shows which plants will be downloaded
- Shows the command that would be run

---

## Step 2: Download Images (Full Run)

Once you're happy with the dry-run output, download images for all missing plants:

```bash
cd gbif_wf_data
python3 tools/download_missing_plants_images.py
```

**What this does:**
1. Identifies all `missing_*` folders in `assets/plant_images/gbif_samples/`
2. Creates a temporary JSON file with only those plants
3. Queries GBIF API for images
4. Downloads images into the corresponding `missing_*` folders
5. Creates metadata files: `index_missing.json` and `index_missing.csv`

**This will take some time** (depends on number of images × internet speed)

---

## Step 3: Customize the Download

Use these options to adjust the behavior:

```bash
# Download up to 30 images per plant (default: 20)
python3 tools/download_missing_plants_images.py --limit-per-plant 30

# Restrict to German occurrences only
python3 tools/download_missing_plants_images.py --restrict-country DE

# Increase candidate pool for better selection
python3 tools/download_missing_plants_images.py --limit 2000

# Combine options
python3 tools/download_missing_plants_images.py \
  --limit-per-plant 25 \
  --preferred-country DE \
  --limit 1500
```

---

## Available Options

```
--limit-per-plant INT       Max images per plant (default: 20)
--min-candidates INT        Min candidates to evaluate (default: 100)
--limit INT                 Total GBIF API limit per search (default: 1000)
--restrict-country CODE     Restrict to country (e.g., DE)
--preferred-country CODE    Prefer country (default: DE)
--dry-run                   Show what would happen, don't download
--help                      Show full help
```

---

## Expected Results

After running, check:

```bash
# View metadata about downloaded images
cat assets/plant_images/gbif_samples/index_missing.json | head -50

# View CSV with all image details
head -20 assets/plant_images/gbif_samples/index_missing.csv

# Check images in a specific plant folder
ls -lh assets/plant_images/gbif_samples/missing_matricaria_recutita/
```

---

## What Gets Downloaded

For each missing plant, the script creates:
- `missing_<plant>/selected/` - Selected high-quality images
- `missing_<plant>/selected_light/` - Lighter versions for thumbnails
- `index_missing.json` - Metadata for all images
- `index_missing.csv` - Spreadsheet-friendly format

---

## Troubleshooting

**"Error: No missing_* folders found"**
- Make sure you're in the correct directory: `cd gbif_wf_data`
- Check that `assets/plant_images/gbif_samples/` exists

**"Connection timeout"**
- GBIF API might be slow, the script retries automatically
- Try with smaller `--limit` value

**"0 plants found in resolved file"**
- Some missing plant names might not match exactly in `plants_resolved_edible.json`
- Check the script output for "not found" warnings

---

## Script Location

```
gbif_wf_data/tools/download_missing_plants_images.py
```

The script automatically:
- Uses `plants_resolved_edible.json` as the source
- Uses `gbif_download_config.json` settings (country: DE, years, etc.)
- Outputs to `assets/plant_images/gbif_samples/`
- Creates metadata in `index_missing.json` and `index_missing.csv`

---

## Next Steps After Download

Once images are downloaded:

1. **Review images** - Check which missing plants now have good image coverage
2. **Run build script** - Use `build_light_release.py` to process images into app format
3. **Update morphology** - Add plant_morphology entries for plants with new images
4. **Test in app** - Rebuild Flutter app to see new images

