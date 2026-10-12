# ELISA image analysis

This folder contains the image-quantification script used for the antibody-array/ELISA analysis in Chapter 4 of the dissertation.

## Manual preprocessing

The assay image must be **cropped and manually divided with a complete red grid before running the script**. The script does **not** detect spot centers and does **not** draw the grid automatically.

For an array with `R` rows and `C` columns, the annotated image should contain `R + 1` horizontal red lines and `C + 1` vertical red lines spanning the full image. The example used here is an 8 x 11 array.

Only the manually gridded image is needed for analysis. The original unannotated image is not used by the script and is therefore not included in this folder.

## Files

- `elisa_grid_quantification.py` — quantitative analysis script.
- `requirements.txt` — Python dependencies.
- `examples/D_grid_example.png` — example input after the grid was drawn manually.
- `examples/D_grid_classified_example.png` — example diagnostic output produced by the script.

## Analysis procedure

The default settings reproduce the analysis used for the Chapter 4 example.

1. The gridded image is converted to grayscale and inverted so that darker ELISA spots have larger numerical values:

   `I = 255 - grayscale`

2. The red grid lines are detected and used only to define the boundaries of each array cell.

3. Within each cell, the lower-left 2 x 2 pixels are used as a local background reference, `I_bg`.

4. The mean of the 10 highest inverted-gray pixel values in the cell is used as a local high-intensity reference, `I_high`.

5. The signal-pixel threshold is defined as

   `I_threshold = I_bg + 0.30 * (I_high - I_bg)`

6. Pixels at or above this threshold are treated as signal pixels. The primary reported value, `signal_mean`, is the mean inverted-gray intensity of these pixels.

The script also exports background-subtracted mean/summed intensities and the number and fraction of pixels classified as signal. These additional values are useful for checking the segmentation but were not required to define the primary signal metric.

The diagnostic output image leaves signal pixels unchanged, marks pixels at or below the local background in light blue, and marks intermediate pixels between background and the signal threshold in green. The diagnostic colors are for visual inspection only and do not enter the numerical calculation.

## Installation

Python 3.10 or later is recommended.

```bash
python -m pip install -r requirements.txt
```

## Usage

From this folder:

```bash
python elisa_grid_quantification.py examples/D_grid_example.png --rows 8 --cols 11 --output-dir output
```

The default analysis parameters can be changed explicitly if needed:

```bash
python elisa_grid_quantification.py IMAGE \
    --rows 8 \
    --cols 11 \
    --background-patch 2 \
    --top-pixels 10 \
    --signal-fraction 0.30 \
    --output-dir output
```

The script checks that the expected number of complete red grid lines can be detected. If the grid is incomplete, the wrong row/column number is supplied, or the line color falls outside the red-grid thresholds, the run stops rather than silently assigning incorrect wells.

## Output

For an input named `D_grid_example.png`, the script writes:

- `D_grid_example_classified.png` — diagnostic classification image.
- `D_grid_example_signals.xlsx` — numerical results.

The Excel workbook contains a `signals` sheet with one row per array cell and a `parameters` sheet recording the analysis settings used for that run.

Important `signals` columns include:

- `well` — array position, such as A1 or B5.
- `background_value` — local background reference.
- `top_pixel_mean` — local high-intensity reference.
- `signal_threshold` — threshold used to define signal pixels.
- `signal_mean` — primary mean signal intensity.
- `signal_mean_bg_subtracted` — mean background-subtracted signal.
- `signal_sum_bg_subtracted` — summed background-subtracted signal.
- `signal_pixels` — number of pixels classified as signal.
- `signal_pixel_fraction` — fraction of usable pixels classified as signal.

## Reproducibility notes

Images compared quantitatively should be acquired under the same exposure/scanner settings. Grid placement is a manual preprocessing step and should be applied consistently across arrays. Saturated pixels cannot be recovered by downstream analysis. The classified diagnostic image should be inspected after each run to verify that the intended spots were captured appropriately.
