# AxoStitch

AxoStitch stitches overlapping 100x optic-nerve microscopy tiles (TIFF) into a single mosaic for axon counting with AxoNet.

It is deliberately **translation-only**: tiles are shifted into place but never rotated, warped or stretched, so axon shapes are not geometrically distorted. Brightness correction is conservative by default: one additive offset per tile, applied equally to all color channels. There is no CLAHE, histogram equalization, sharpening or denoising.

## How it works

1. Reads a ZIP archive or a folder of `.tif` / `.tiff` tiles.
2. Rejects unreadable and blank tiles, and skips exact duplicates.
3. Matches overlapping tiles with SIFT at reduced resolution.
4. Solves for globally consistent tile positions with weighted least squares.
5. Refuses to continue if the tiles don't form one connected mosaic (unless told otherwise; see `--allow-small-disconnected`).
6. Evens out brightness between tiles using their overlaps, then feather-blends the seams.
7. Writes the full-resolution mosaic, a preview, QC files and an optional AxoNet-scale copy.

## Requirements

- Python 3.11 (other recent 3.x versions should work)
- The packages in `AxoStitch/requirements.txt`: NumPy, OpenCV and tifffile

## Setup

Run these in PowerShell from the repository folder. You only need to do this once.

```powershell
cd AxoStitch
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

After activation, the prompt starts with `(.venv)`. In later sessions, you only need to `cd AxoStitch` and run `.venv\Scripts\Activate.ps1` again.

> If PowerShell refuses to run `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

## Usage

```powershell
python AxoStitch.py "path\to\tiles.zip" `
    --sid "2003 OS" `
    --output-dir "..\outputs\2003 OS" `
    --axon-scale 0.542 `
    --allow-small-disconnected
```

The input can be a ZIP file or a folder of TIFFs. The backtick (`` ` ``) at the end of a line continues the command onto the next line; you can also type it all on one line.

AxoStitch will not overwrite existing results. If the output files already exist, it stops. Use a separate output folder per sample, or move the old results first.

### Options

| Option | Default | Description |
|---|---|---|
| `input` | *(required)* | ZIP archive or folder containing the TIFF tiles |
| `--sid` | *(required)* | Sample ID, used to name the output files, e.g. `"2003 OS"` |
| `--output-dir` | `stitched_output` | Folder for the results |
| `--axon-scale` | off | Also write a downscaled copy for AxoNet, e.g. `0.542` |
| `--allow-small-disconnected` | off | If at most 2 files don't connect to the mosaic and at least 90% of tiles do, drop those files and continue. Useful when overview or reference images are in the same folder. |
| `--photometric-correction` | `brightness` | `brightness` (one offset per tile, recommended for AxoNet), `rgb` (per-channel), or `none` |
| `--max-brightness-offset` | `35.0` | Largest brightness correction allowed, on the 0–255 scale |
| `--no-feather` | off | Disable feather blending at the seams |
| `--keep-duplicates` | off | Keep tiles that are exact duplicates of another tile |
| `--jpeg-quality` | `92` | Quality of the full-resolution JPEG |
| `--preview-max` | `2600` | Maximum width/height of the preview image, in pixels |

#### Registration tuning

You rarely need to change these. Try them if tiles fail to connect.

| Option | Default | Description |
|---|---|---|
| `--registration-scale` | `0.15` | Downscale factor used for feature matching |
| `--sift-features` | `4000` | Maximum SIFT features per tile |
| `--ratio-test` | `0.72` | Lowe ratio threshold for SIFT matches |
| `--min-good-matches` | `12` | Matches needed before a tile pair is considered |
| `--min-inliers` | `10` | Consistent matches needed to link two tiles |
| `--inlier-radius` | `3.0` | Allowed match disagreement, in pixels at registration scale |

Run `python AxoStitch.py --help` for the full list.

## Output

All files start with the sample ID (`<SID>` below).

| File | Contents |
|---|---|
| `<SID>_100x_stitched_100pct_CLASSIC.tif` | Full-resolution mosaic (classic TIFF, not BigTIFF) |
| `<SID>_100x_stitched_100pct.jpg` | Full-resolution JPEG |
| `<SID>_100x_preview.jpg` | Downscaled preview for checking the result |
| `<SID>_tile_QC.csv` | Per-tile checks: blank, duplicate, unreadable |
| `<SID>_tile_positions.csv` | Final position of each tile in the mosaic |
| `<SID>_registration_edges.csv` | Every tile-to-tile match used to build the mosaic |
| `<SID>_100x_<scale>x_AXONET_CLASSIC.tif` / `.jpg` | AxoNet-scale copy (only with `--axon-scale`) |

## Before you count axons

**Always open the preview image and check it before using a mosaic in AxoNet or any other counting software.**

The program should report that nearly all 100x tiles are connected. If several tiles are disconnected, or it stops with a registration error, do not use that mosaic until you've found out why. Common causes are non-overlapping tiles, out-of-focus fields, or unrelated images in the input folder.
