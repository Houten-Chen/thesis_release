#!/usr/bin/env python3
"""Batch resolution test for RHP EHMA surface-patch analysis.

Place this file, rhp_surface_patches.py, and the PDB files in the same folder,
then run:

    python rhp_resolution_test.py

Default grid spacings: 0.50, 0.35, 0.25 A.
The script reuses the geometry/patch functions from rhp_surface_patches.py but
writes only compact validation outputs (no meshes, atom tables, or per-run
analysis folders).

Main outputs in ./resolution_test_output/:
    resolution_metrics.csv     one row per PDB x grid spacing
    patch_areas_long.csv       one row per detected EHMA patch
    resolution_pairwise.csv    per-PDB comparison to the finest spacing
    resolution_summary.csv     aggregate statistics across all PDBs
    failures.csv               any failed PDB/resolution runs
    run_config.json            parameters and file counts

The resolution-selection idea is to look for a plateau: decreasing grid spacing
should stop materially changing the patch-area distribution, especially the
small-patch tail. Do NOT choose a spacing merely because it produces the fewest
small patches.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance


DEFAULT_RESOLUTIONS = (0.50, 0.35, 0.25)
DEFAULT_SMALL_THRESHOLDS = (2.0, 5.0, 10.0, 20.0)


def load_core(script_path: Path):
    spec = importlib.util.spec_from_file_location("rhp_surface_patches_core", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        if rows:
            columns = list(rows[0].keys())
        else:
            columns = []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def safe_mean(x):
    return float(np.mean(x)) if len(x) else None


def safe_median(x):
    return float(np.median(x)) if len(x) else None


def safe_sd(x):
    return float(np.std(x, ddof=1)) if len(x) >= 2 else (0.0 if len(x) == 1 else None)


def safe_var(x):
    return float(np.var(x, ddof=1)) if len(x) >= 2 else (0.0 if len(x) == 1 else None)


def safe_quantile(x, q):
    return float(np.quantile(x, q)) if len(x) else None


def safe_cv(x):
    if len(x) < 2:
        return 0.0 if len(x) == 1 else None
    m = float(np.mean(x))
    return float(np.std(x, ddof=1) / m) if m > 0 else None


def rel_change(value, ref):
    if value is None or ref is None:
        return None
    if ref == 0:
        return 0.0 if value == 0 else None
    return float((value - ref) / ref)


def summarize_patch_areas(
    areas: np.ndarray,
    ehma_total_area: float,
    thresholds: tuple[float, ...],
) -> dict:
    areas = np.asarray(areas, dtype=float)
    out = {
        "patch_count": int(len(areas)),
        "patch_area_mean_A2": safe_mean(areas),
        "patch_area_median_A2": safe_median(areas),
        "patch_area_sd_A2": safe_sd(areas),
        "patch_area_variance_A4": safe_var(areas),
        "patch_area_cv": safe_cv(areas),
        "patch_area_q10_A2": safe_quantile(areas, 0.10),
        "patch_area_q25_A2": safe_quantile(areas, 0.25),
        "patch_area_q75_A2": safe_quantile(areas, 0.75),
        "patch_area_q90_A2": safe_quantile(areas, 0.90),
        "largest_patch_A2": float(areas.max()) if len(areas) else 0.0,
        "smallest_patch_A2": float(areas.min()) if len(areas) else None,
        "area_weighted_mean_A2": (
            float(np.dot(areas, areas) / areas.sum())
            if len(areas) and areas.sum() > 0 else None
        ),
    }

    for threshold in thresholds:
        tag = str(int(threshold)) if float(threshold).is_integer() else str(threshold).replace(".", "p")
        mask = areas < threshold
        small_area = float(areas[mask].sum()) if len(areas) else 0.0
        out[f"patch_count_lt_{tag}_A2"] = int(np.count_nonzero(mask))
        out[f"patch_count_fraction_lt_{tag}_A2"] = (
            float(np.count_nonzero(mask) / len(areas)) if len(areas) else 0.0
        )
        out[f"patch_area_sum_lt_{tag}_A2"] = small_area
        out[f"patch_area_fraction_lt_{tag}_A2"] = (
            float(small_area / ehma_total_area) if ehma_total_area > 0 else 0.0
        )
    return out


def analyze_one(core, pdb: Path, spacing: float, args, thresholds):
    atoms, warnings = core.read_pdb(pdb, core.RESIDUE_MAP, args.model)
    xyz = np.array([a.xyz for a in atoms], dtype=np.float64)
    residues, max_bond = core.classify_and_check(atoms, set(args.ehma_carbons.split(",")))
    atomic_radii = np.array([core.VDW_RADII[a.element] for a in atoms], dtype=float)
    expanded = atomic_radii + args.probe_radius

    vertices, faces, vertex_area, grid_info = core.make_sas(
        xyz,
        expanded,
        spacing,
        args.max_grid_cells,
        args.include_cavities,
    )
    owner, surface_error = core.assign_owner(vertices, xyz, expanded)
    atom_area = np.bincount(owner, weights=vertex_area, minlength=len(atoms))
    _, patches = core.find_ehma_patches(vertices, faces, vertex_area, owner, atoms)

    total_area = float(vertex_area.sum())
    ehma_area = float(sum(atom_area[i] for i, a in enumerate(atoms) if a.ehma_alkyl))
    patch_areas = np.array([p["area_A2"] for p in patches], dtype=float)

    if not np.isclose(patch_areas.sum() if len(patch_areas) else 0.0,
                      ehma_area, rtol=1e-10, atol=1e-7):
        raise RuntimeError("EHMA patch areas do not conserve EHMA alkyl surface area")

    metrics = {
        "pdb": pdb.name,
        "grid_spacing_A": float(spacing),
        "n_atoms": len(atoms),
        "n_repeat_units": len(residues),
        "total_surface_A2": total_area,
        "EHMA_alkyl_surface_A2": ehma_area,
        "EHMA_alkyl_surface_fraction": ehma_area / total_area if total_area > 0 else None,
        "max_vertex_distance_from_exact_SAS_A": surface_error,
        "max_backbone_bond_A": max_bond,
        "grid_cells": int(grid_info["grid_cells"]),
        "mesh_vertices": int(grid_info["vertices"]),
        "mesh_triangles": int(grid_info["triangles"]),
        "removed_cavity_grid_cells": int(grid_info["removed_cavity_grid_cells"]),
        "warnings": " | ".join(warnings),
    }
    metrics.update(summarize_patch_areas(patch_areas, ehma_area, thresholds))

    patch_rows = []
    for p in patches:
        patch_rows.append({
            "pdb": pdb.name,
            "grid_spacing_A": float(spacing),
            "patch_id": int(p["patch_id"]),
            "patch_area_A2": float(p["area_A2"]),
            "patch_area_nm2": float(p["area_nm2"]),
            "n_EHMA_units": int(p["n_EHMA_units"]),
            "n_vertices": int(p["n_vertices"]),
        })

    # Explicitly release the largest arrays before moving to the next job.
    del vertices, faces, vertex_area, owner, atom_area, xyz, atomic_radii, expanded
    gc.collect()
    return metrics, patch_rows, patch_areas


def aggregate_resolution_summary(metrics_rows, thresholds):
    rows = []
    spacings = sorted({float(r["grid_spacing_A"]) for r in metrics_rows}, reverse=True)
    metric_names = [
        "patch_count",
        "patch_area_mean_A2",
        "patch_area_median_A2",
        "patch_area_sd_A2",
        "patch_area_variance_A4",
        "patch_area_cv",
        "patch_area_q25_A2",
        "smallest_patch_A2",
        "largest_patch_A2",
        "area_weighted_mean_A2",
        "EHMA_alkyl_surface_A2",
        "EHMA_alkyl_surface_fraction",
    ]
    for threshold in thresholds:
        tag = str(int(threshold)) if float(threshold).is_integer() else str(threshold).replace(".", "p")
        metric_names.extend([
            f"patch_count_lt_{tag}_A2",
            f"patch_count_fraction_lt_{tag}_A2",
            f"patch_area_fraction_lt_{tag}_A2",
        ])

    for spacing in spacings:
        subset = [r for r in metrics_rows if float(r["grid_spacing_A"]) == spacing]
        out = {"grid_spacing_A": spacing, "n_PDB": len(subset)}
        for name in metric_names:
            vals = np.array([float(r[name]) for r in subset if r.get(name) not in (None, "")], dtype=float)
            out[name + "__mean_across_PDB"] = safe_mean(vals)
            out[name + "__sd_across_PDB"] = safe_sd(vals)
            out[name + "__median_across_PDB"] = safe_median(vals)
        rows.append(out)
    return rows


def make_pairwise_rows(metrics_rows, patch_area_map, reference_spacing):
    by_pdb = {}
    for row in metrics_rows:
        by_pdb.setdefault(row["pdb"], {})[float(row["grid_spacing_A"])] = row

    rows = []
    compare_fields = [
        "patch_count",
        "patch_area_mean_A2",
        "patch_area_median_A2",
        "patch_area_sd_A2",
        "patch_area_variance_A4",
        "patch_area_cv",
        "patch_area_q25_A2",
        "smallest_patch_A2",
        "largest_patch_A2",
        "area_weighted_mean_A2",
        "EHMA_alkyl_surface_A2",
    ]

    for pdb, spacing_rows in sorted(by_pdb.items()):
        if reference_spacing not in spacing_rows:
            continue
        ref = spacing_rows[reference_spacing]
        ref_areas = patch_area_map.get((pdb, reference_spacing), np.array([], dtype=float))
        for spacing, row in sorted(spacing_rows.items(), reverse=True):
            if spacing == reference_spacing:
                continue
            areas = patch_area_map.get((pdb, spacing), np.array([], dtype=float))
            out = {
                "pdb": pdb,
                "grid_spacing_A": spacing,
                "reference_spacing_A": reference_spacing,
                "wasserstein_patch_area_A2": (
                    float(wasserstein_distance(areas, ref_areas))
                    if len(areas) and len(ref_areas) else None
                ),
            }
            for field in compare_fields:
                value = row.get(field)
                ref_value = ref.get(field)
                out[field + "__value"] = value
                out[field + "__reference"] = ref_value
                out[field + "__relative_change"] = rel_change(value, ref_value)
            rows.append(out)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Batch-test grid resolution for RHP EHMA patch-size distributions."
    )
    parser.add_argument(
        "--folder", type=Path, default=Path.cwd(),
        help="Folder containing PDBs and rhp_surface_patches.py (default: current folder)"
    )
    parser.add_argument(
        "--core-script", type=Path, default=None,
        help="Path to rhp_surface_patches.py (default: <folder>/rhp_surface_patches.py)"
    )
    parser.add_argument(
        "--pattern", default="*.pdb", help='PDB glob pattern (default: "*.pdb")'
    )
    parser.add_argument(
        "--resolutions", nargs="+", type=float, default=list(DEFAULT_RESOLUTIONS),
        help="Grid spacings in Angstrom (default: 0.50 0.35 0.25)"
    )
    parser.add_argument(
        "--small-thresholds", nargs="+", type=float, default=list(DEFAULT_SMALL_THRESHOLDS),
        help="Patch-area thresholds in A^2 for the small-patch tail (default: 2 5 10 20)"
    )
    parser.add_argument(
        "--probe-radius", type=float, default=1.4,
        help="SAS probe radius in Angstrom (default: 1.4)"
    )
    parser.add_argument(
        "--ehma-carbons", default=",".join(f"C{i}" for i in range(7, 15)),
        help="Comma-separated EHMA alkyl carbon names (default: C7...C14)"
    )
    parser.add_argument("--model", type=int, default=1, help="1-based MODEL occurrence (default: 1)")
    parser.add_argument(
        "--max-grid-cells", type=int, default=40_000_000,
        help="Maximum grid cells per analysis (default: 40000000)"
    )
    parser.add_argument(
        "--include-cavities", action="store_true",
        help="Include sealed internal cavity surfaces (default: off)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: <folder>/resolution_test_output)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional limit for a quick test, e.g. --limit 10"
    )
    args = parser.parse_args()

    args.folder = args.folder.resolve()
    if args.core_script is None:
        args.core_script = args.folder / "rhp_surface_patches.py"
    else:
        args.core_script = args.core_script.resolve()
    if args.output_dir is None:
        args.output_dir = args.folder / "resolution_test_output"
    else:
        args.output_dir = args.output_dir.resolve()

    if not args.core_script.is_file():
        parser.error(f"Core script not found: {args.core_script}")
    if any((not math.isfinite(x) or x <= 0) for x in args.resolutions):
        parser.error("All --resolutions values must be finite and > 0")
    if any((not math.isfinite(x) or x < 0) for x in args.small_thresholds):
        parser.error("All --small-thresholds values must be finite and >= 0")

    resolutions = tuple(sorted(set(float(x) for x in args.resolutions), reverse=True))
    thresholds = tuple(sorted(set(float(x) for x in args.small_thresholds)))
    reference_spacing = min(resolutions)

    pdbs = sorted(args.folder.glob(args.pattern))
    if args.limit is not None:
        pdbs = pdbs[:args.limit]
    if not pdbs:
        parser.error(f"No PDB files matched {args.pattern!r} in {args.folder}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    core = load_core(args.core_script)

    print(f"Found {len(pdbs)} PDB files.")
    print("Grid spacings: " + ", ".join(f"{x:g} A" for x in resolutions))
    print(f"Reference spacing for pairwise comparison: {reference_spacing:g} A")

    metrics_rows = []
    patch_rows = []
    failures = []
    patch_area_map = {}
    started_all = time.perf_counter()

    total_jobs = len(pdbs) * len(resolutions)
    job_index = 0
    for pdb in pdbs:
        for spacing in resolutions:
            job_index += 1
            started = time.perf_counter()
            try:
                metrics, patches, areas = analyze_one(core, pdb, spacing, args, thresholds)
                metrics["elapsed_seconds"] = time.perf_counter() - started
                metrics_rows.append(metrics)
                patch_rows.extend(patches)
                patch_area_map[(pdb.name, spacing)] = areas
                print(
                    f"[{job_index:4d}/{total_jobs}] {pdb.name}  {spacing:.2f} A  "
                    f"patches={metrics['patch_count']:3d}  "
                    f"small<10A2={metrics.get('patch_area_fraction_lt_10_A2', 0.0):.3f}  "
                    f"median={metrics['patch_area_median_A2'] if metrics['patch_area_median_A2'] is not None else float('nan'):.2f} A2",
                    flush=True,
                )
            except Exception as exc:  # keep the batch alive and report every failure
                failures.append({
                    "pdb": pdb.name,
                    "grid_spacing_A": spacing,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                print(
                    f"[{job_index:4d}/{total_jobs}] FAILED {pdb.name} {spacing:.2f} A: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

    summary_rows = aggregate_resolution_summary(metrics_rows, thresholds) if metrics_rows else []
    pairwise_rows = make_pairwise_rows(metrics_rows, patch_area_map, reference_spacing) if metrics_rows else []

    write_csv(args.output_dir / "resolution_metrics.csv", metrics_rows)
    write_csv(args.output_dir / "patch_areas_long.csv", patch_rows)
    write_csv(args.output_dir / "resolution_pairwise.csv", pairwise_rows)
    write_csv(args.output_dir / "resolution_summary.csv", summary_rows)
    write_csv(
        args.output_dir / "failures.csv",
        failures,
        ["pdb", "grid_spacing_A", "error_type", "error"],
    )

    config = {
        "folder": str(args.folder),
        "core_script": str(args.core_script),
        "core_script_version": getattr(core, "VERSION", None),
        "pdb_pattern": args.pattern,
        "n_PDB_selected": len(pdbs),
        "resolutions_A": list(resolutions),
        "reference_spacing_A": reference_spacing,
        "small_patch_thresholds_A2": list(thresholds),
        "probe_radius_A": args.probe_radius,
        "EHMA_carbons": args.ehma_carbons,
        "model_occurrence": args.model,
        "include_cavities": args.include_cavities,
        "max_grid_cells": args.max_grid_cells,
        "n_successful_jobs": len(metrics_rows),
        "n_failed_jobs": len(failures),
        "elapsed_seconds": time.perf_counter() - started_all,
        "interpretation_note": (
            "Choose a grid spacing from a convergence plateau, not from the spacing "
            "that simply minimizes small-patch counts. Compare patch-area distributions, "
            "small-patch area fractions, and pairwise Wasserstein distances to the finest spacing."
        ),
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nFinished.")
    print(f"Successful jobs: {len(metrics_rows)}/{total_jobs}; failures: {len(failures)}")
    print(f"Output folder: {args.output_dir}")
    print("Please return at least: resolution_summary.csv, resolution_pairwise.csv, "
          "resolution_metrics.csv, patch_areas_long.csv, and failures.csv.")


if __name__ == "__main__":
    main()
