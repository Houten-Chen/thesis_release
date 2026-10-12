#!/usr/bin/env python3
"""Quantify an ELISA array image carrying a manually drawn red grid.

The defaults preserve the original 11-column, 8-row calculation. This script
locates existing grid lines; grid drawing and spot detection are manual steps.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import string
import sys

import cv2
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SKY_BLUE_BGR = (235, 206, 135)
GREEN_BGR = (0, 255, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image", nargs="?", type=Path, default=SCRIPT_DIR / "examples" / "D_grid_example.png",
        help="8-bit color PNG/TIFF with a complete, manually drawn red grid",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("elisa_results"))
    parser.add_argument("--cols", type=int, default=11, help="number of grid columns")
    parser.add_argument("--rows", type=int, default=8, help="number of grid rows (1-26)")
    parser.add_argument("--bg-size", "--background-patch", dest="bg_size", type=int, default=2, help="side of lower-left background square")
    parser.add_argument("--top-n", "--top-pixels", dest="top_n", type=int, default=10, help="number of brightest inverted pixels used for the upper reference")
    parser.add_argument("--threshold-fraction", "--signal-fraction", dest="threshold_fraction", type=float, default=0.30, help="fraction between background and upper reference")
    parser.add_argument("--preview", action="store_true", help="show the classified image after saving")
    args = parser.parse_args()
    if args.cols < 1 or not 1 <= args.rows <= 26:
        parser.error("--cols must be positive and --rows must be between 1 and 26")
    if args.bg_size < 1 or args.top_n < 1:
        parser.error("--bg-size and --top-n must be positive")
    if not 0 <= args.threshold_fraction <= 1:
        parser.error("--threshold-fraction must be between 0 and 1")
    return args


def read_grid_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise ValueError(f"Input image does not exist: {path}")
    # Reading bytes first also supports paths containing non-ASCII characters.
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot decode input image: {path}")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("Input must be an 8-bit color image with a manually drawn red grid.")
    return image[:, :, :3].copy()  # OpenCV BGR; ignore an optional alpha channel.


def find_cell_bounds(red_mask: np.ndarray, cx: int, cy: int, well: str) -> tuple[int, int, int, int]:
    """Scan from the expected cell center to existing red lines, as in the original."""
    height, width = red_mask.shape
    if red_mask[cy, cx]:
        raise ValueError(f"{well}: expected cell center lies on a red line; check crop and grid dimensions.")
    top = bottom = cy
    left = right = cx
    while top > 0 and not red_mask[top, cx]:
        top -= 1
    while bottom < height - 1 and not red_mask[bottom, cx]:
        bottom += 1
    while left > 0 and not red_mask[cy, left]:
        left -= 1
    while right < width - 1 and not red_mask[cy, right]:
        right += 1
    boundaries = (("top", top, cx), ("bottom", bottom, cx),
                  ("left", cy, left), ("right", cy, right))
    missing = [side for side, y, x in boundaries if not red_mask[y, x]]
    if missing:
        raise ValueError(f"{well}: missing red boundary ({', '.join(missing)}); complete the grid manually.")
    # x2 and y2 are exclusive. The surrounding red lines are outside this ROI.
    x1, y1, x2, y2 = left + 1, top + 1, right, bottom
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{well}: grid encloses an empty cell.")
    if red_mask[y1:y2, x1:x2].any():
        raise ValueError(f"{well}: red grid pixels enter the measurement region; straighten the grid manually.")
    return x1, y1, x2, y2


def analyze(image: np.ndarray, rows: int = 8, cols: int = 11, bg_size: int = 2,
            top_n: int = 10, threshold_fraction: float = 0.30) -> tuple[pd.DataFrame, np.ndarray]:
    height, width = image.shape[:2]
    gray_inv = 255 - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    red_mask = (image[:, :, 2] > 180) & (image[:, :, 1] < 80) & (image[:, :, 0] < 80)
    if not red_mask.any():
        raise ValueError("No red grid found. Draw the grid manually before running this script.")
    classified = image.copy()
    results = []
    seen_bounds = set()
    for j in range(rows):
        for i in range(cols):
            well = f"{string.ascii_uppercase[j]}{i + 1}"
            cx, cy = int((i + 0.5) * width / cols), int((j + 0.5) * height / rows)
            x1, y1, x2, y2 = find_cell_bounds(red_mask, cx, cy, well)
            bounds = (x1, y1, x2, y2)
            if bounds in seen_bounds:
                raise ValueError(f"{well}: another cell has the same bounds; check --rows and --cols.")
            seen_bounds.add(bounds)
            region = gray_inv[y1:y2, x1:x2]
            if min(region.shape) < bg_size or region.size <= bg_size ** 2:
                raise ValueError(f"{well}: cell is too small for the background patch.")
            # The background patch is included in the upper reference, matching
            # the original script, but excluded from all pixel classifications.
            background = np.zeros(region.shape, dtype=bool)
            background[-bg_size:, :bg_size] = True
            bg_value = float(np.mean(region[background]))
            highest = np.sort(region, axis=None)[-top_n:]
            upper_reference = float(np.mean(highest))
            threshold = bg_value + threshold_fraction * (upper_reference - bg_value)
            candidates = ~background
            type1 = candidates & (region <= bg_value)
            type2 = candidates & (region > bg_value) & (region < threshold)
            signal = candidates & (region > bg_value) & (region >= threshold)
            colored_region = classified[y1:y2, x1:x2]
            colored_region[type1] = SKY_BLUE_BGR
            colored_region[type2] = GREEN_BGR
            signal_mean = float(np.mean(region[signal])) if signal.any() else 0.0
            results.append({
                "well": well,
                "bg_threshold": bg_value,
                "green_threshold": threshold,
                "signal_mean": signal_mean,
                "signal_pixels": int(signal.sum()),
                "total_pixels": int(candidates.sum()),
                "top_pixels_mean": upper_reference,
                "x1": x1, "y1": y1, "x2_exclusive": x2, "y2_exclusive": y2,
            })
    return pd.DataFrame(results), classified


def save_results(df: pd.DataFrame, classified: np.ndarray, args: argparse.Namespace) -> list[Path]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    csv_path = output_dir / f"{stem}_signals.csv"
    xlsx_path = output_dir / f"{stem}_signals.xlsx"
    png_path = output_dir / f"{stem}_classified.png"
    json_path = output_dir / f"{stem}_analysis.json"
    df.to_csv(csv_path, index=False)
    matrix = pd.DataFrame(df["signal_mean"].to_numpy().reshape(args.rows, args.cols),
                          index=list(string.ascii_uppercase[:args.rows]),
                          columns=list(range(1, args.cols + 1)))
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="signals", index=False)
        matrix.to_excel(writer, sheet_name="signal_mean", index_label="row")
    success, encoded = cv2.imencode(".png", classified)
    if not success:
        raise ValueError("Could not encode the classified image.")
    encoded.tofile(png_path)
    metadata = {
        "input_file": args.image.name,
        "input_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "image_height": int(classified.shape[0]), "image_width": int(classified.shape[1]),
        "rows": args.rows, "cols": args.cols, "background_patch_side_pixels": args.bg_size,
        "top_n": args.top_n, "threshold_fraction": args.threshold_fraction,
        "red_grid_rule_rgb": "R > 180, G < 80, B < 80",
        "signal_definition": "Mean of retained 8-bit inverted grayscale pixels; no background subtraction or normalization.",
        "versions": {"python": sys.version.split()[0], "opencv": cv2.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__,
                     "openpyxl": importlib.metadata.version("openpyxl")},
        "output_files": [csv_path.name, xlsx_path.name, png_path.name, json_path.name],
    }
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return [csv_path, xlsx_path, png_path, json_path]


def main() -> int:
    args = parse_args()
    try:
        image = read_grid_image(args.image)
        df, classified = analyze(image, args.rows, args.cols, args.bg_size,
                                 args.top_n, args.threshold_fraction)
        paths = save_results(df, classified, args)
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Analyzed {len(df)} cells ({args.rows} rows x {args.cols} columns).")
    for path in paths:
        print(f"Saved: {path}")
    if args.preview:
        import matplotlib.pyplot as plt
        plt.imshow(cv2.cvtColor(classified, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Sky blue: at/below background; green: below signal threshold")
        plt.tight_layout()
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
