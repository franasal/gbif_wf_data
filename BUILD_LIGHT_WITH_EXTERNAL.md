# Building Light Assets with External Storage

## Overview

When you've moved `reviewed_*` folders to external storage to save disk space, you need a way to rebuild the light asset packages without having all folders locally. This script handles that scenario.

## What the Script Does

The `build_light_external.py` script:
1. **Combines folders** from local and external storage using symlinks
2. **Skips missing_* folders** (by default) to focus on reviewed content
3. **Calls the standard build process** with the combined tree
4. **Creates light asset ZIPs** automatically

---

## Quick Start

### Step 1: Locate Your External Storage

First, find where your reviewed_ folders are mounted or copied:

```bash
# Example: External drive mounted at /mnt/external
ls -la /mnt/external/gbif_samples/reviewed_* | head -5
```

Or if copied to a specific path:
```bash
ls -la /path/to/external/storage/gbif_samples/
```

### Step 2: Test with Dry Run

```bash
cd gbif_wf_data

python3 tools/build_light_external.py \
    --external-root /mnt/external/gbif_samples \
    --dry-run
```

**Output will show:**
- How many local folders found
- How many external folders found
- Total folders that will be processed
- The command that would be executed

### Step 3: Run the Build

Once satisfied with the dry-run output:

```bash
cd gbif_wf_data

python3 tools/build_light_external.py \
    --external-root /mnt/external/gbif_samples
```

**This will:**
1. Create temporary symlinks combining local + external
2. Run the build process on the combined tree
3. Generate light asset ZIPs
4. Automatically clean up temporary files

---

## Command Options

```bash
python3 tools/build_light_external.py \
    --external-root PATH          # Required: path to external storage
    [--local-root PATH]           # Local gbif_samples (default: ./assets/plant_images/gbif_samples)
    [--skip-missing]              # Exclude missing_* folders (default: enabled)
    [--include-missing]           # Include missing_* folders
    [--split-by-class]            # Split output to edible/poisonous
    [--keep-prefixes]             # Keep reviewed_/missing_ in paths (default: enabled)
    [--strip-prefixes]            # Remove prefixes from paths
    [--max-images-per-plant N]    # Limit images per plant (0 = unlimited)
    [--max-width N]               # Max image width (default: 1024)
    [--max-long-edge N]           # Max long edge (default: 2048)
    [--quality N]                 # WebP quality 1-100 (default: 85)
    [--dry-run]                   # Show what would happen
    [--help]                      # Show full help
```

---

## Common Usage Patterns

### Build Everything (Default)
```bash
python3 tools/build_light_external.py \
    --external-root /mnt/external/gbif_samples
```
- Uses all reviewed_* and has_lookalike_* folders
- Skips missing_* folders
- Keeps prefixes in output for curation tracking
- Combines into single ZIP

---

### Build Edible and Poisonous Separately
```bash
python3 tools/build_light_external.py \
    --external-root /mnt/external/gbif_samples \
    --split-by-class
```
- Creates two ZIPs: edible.zip and poisonous.zip
- Requires `names_edible.json` and `names_poisonous.json`
- Better for selective app deployment

---

### Higher Quality Images
```bash
python3 tools/build_light_external.py \
    --external-root /mnt/external/gbif_samples \
    --quality 90 \
    --max-width 1280 \
    --max-long-edge 2560
```
- WebP quality 90 (more detail, larger files)
- Larger image dimensions
- Slightly slower build, better visual quality

---

### Smaller File Size
```bash
python3 tools/build_light_external.py \
    --external-root /mnt/external/gbif_samples \
    --quality 75 \
    --max-width 800 \
    --max-images-per-plant 10
```
- Lower quality (faster, smaller)
- Smaller image dimensions
- Limit to 10 best images per plant
- Good for mobile/web with limited bandwidth

---

### Include Missing Folders Too
```bash
python3 tools/build_light_external.py \
    --external-root /mnt/external/gbif_samples \
    --include-missing
```
- Includes missing_* folders in addition to reviewed_*
- Useful when missing plants get new images from GBIF downloads

---

## Understanding the Output

After successful build, you'll see:

```
[output_root] Images: 12345 | 456.78 MB
Manifest JSON: gbif_wf_data/assets/plant_images/light_build/images_manifest.json
Manifest CSV: gbif_wf_data/assets/plant_images/light_build/images_manifest.csv
ZIP: gbif_wf_data/assets/plant_images/light_build/light_pack.zip
```

### Files Created
- **light_pack.zip** - The main asset pack for the app
- **images_manifest.json** - Metadata about all images
- **images_manifest.csv** - Same metadata in spreadsheet format
- **Webp files** - Individual processed images in plant folders

---

## Folder Structure Explanation

Before running:
```
Local Storage (fast, limited space):
  gbif_wf_data/assets/plant_images/gbif_samples/
    └─ missing_*/ (kept locally)
    └─ has_lookalike_*/
    └─ looked_similar_*/

External Storage (slow, plenty of space):
  /mnt/external/gbif_samples/
    └─ reviewed_*/  (all reviewed folders)
    └─ reviewed_has_lookalike_*/
```

What the script does:
```
Creates temporary combined tree (symlinked):
  /tmp/temp_dir/combined/
    ├─ missing_* → local symlink
    ├─ has_lookalike_* → local symlink
    ├─ reviewed_* → external symlink
    ├─ reviewed_has_lookalike_* → external symlink
    └─ ... (all others)
```

Then builds light assets from this combined tree.

---

## Troubleshooting

### "External root not found"
```bash
# Check the path is correct and mounted
ls -la /mnt/external/gbif_samples/

# Or find it
find /mnt -name "gbif_samples" -type d 2>/dev/null
```

### "0 folders found in external"
- Check path has reviewed_* folders
- Check permissions: `ls -la /mnt/external/gbif_samples/ | head`

### Build is very slow
- External storage might be slow (USB, network drive)
- Consider copying frequently-used folders to local first
- Or increase `--max-images-per-plant` to process fewer

### Symlink permission errors
- Check you have read access to both local and external
- Check temporary directory has write permissions

---

## Integration with Workflow

After building light assets:

1. **Verify the ZIP**
   ```bash
   unzip -l gbif_wf_data/assets/plant_images/light_build/light_pack.zip | head -20
   ```

2. **Check image count**
   ```bash
   jq '.images_count' gbif_wf_data/assets/plant_images/light_build/images_manifest.json
   ```

3. **Update app**
   - Copy light_pack.zip to app assets
   - Rebuild Flutter app
   - Test on device

4. **Archive results** (optional)
   ```bash
   cp gbif_wf_data/assets/plant_images/light_build/light_pack.zip \
      ~/Archive/light_pack_$(date +%Y%m%d).zip
   ```

---

## Disk Space Savings

Example with 23 reviewed_* folders (~50 GB):
- **Before**: All folders on SSD → 50 GB wasted
- **After**: Only missing_* local + external reviewed_* → <5 GB local

When you need to rebuild:
1. External drive gets mounted
2. Script combines using symlinks (no copying)
3. Build happens in temp directory
4. Results stay on local SSD

---

## Next Steps

After building light assets:

1. **Test in app** - Rebuild Flutter and verify images show
2. **Add morphology** - For newly downloaded plants, add morphology data
3. **Update app version** - Increment version in pubspec.yaml
4. **Commit changes** - Save updated light_pack.zip

---

## Questions?

Check the main script help:
```bash
python3 tools/build_light_external.py --help
```

Or review the original build script:
```bash
python3 tools/build_light_release.py --help
```
