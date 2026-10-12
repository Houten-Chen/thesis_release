"""Window-level HLB/Q3 statistics on an explicitly supplied PCA reference."""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .common import (Grid, HLB, check_embedding_metadata, encode_sequence, input_records,
                     latent_rows, load_grid_table, load_pca, meta_path, prepare_outputs,
                     project, read_latents, read_table, save_json, sha256, sliding_positions)


def lag_profile(values, max_lag=30):
    values = np.asarray(values, dtype=float)
    if not 1 <= max_lag < len(values):
        raise ValueError("max_lag must be between 1 and window length - 1")
    # Keep the arithmetic and lag order in the uploaded calc_lag_diffs function.
    return np.array([np.mean([abs(values[i] - values[i + lag]) for i in range(len(values) - lag)])
                     for lag in range(1, max_lag + 1)])


def lag_entropy(profile, eps=1e-12):
    p = np.abs(profile)
    total = np.nansum(p)
    if total == 0 or np.isnan(total):
        return np.nan
    p = p / total
    return -np.nansum(p * np.log(p + eps))


def q3_fraction(q3):
    if not q3 or set(q3) - set("HEC"):
        raise ValueError("Q3 strings must contain only H, E, and C")
    return np.array([q3.count(code) / len(q3) for code in "HEC"])


def structure_tendency(fractions):
    values = np.asarray(fractions, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * np.log((values[..., 0] + values[..., 1]) / (values[..., 2] + 1e-12))


def fit_pca(paths, output, n_components=5, overwrite=False):
    if not 2 <= n_components <= 16:
        raise ValueError("PCA component count must be between 2 and 16")
    checkpoint = check_embedding_metadata(paths)
    values = np.concatenate([read_latents(path)[0] for path in paths])
    if len(values) < n_components:
        raise ValueError("Too few latent vectors for the requested PCA component count")
    prepare_outputs([output], overwrite, paths)
    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(values)
    info = {"mean": pca.mean_.tolist(), "components": pca.components_.tolist(),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "n_reference_windows": len(values), "reference_inputs": input_records(paths),
            "checkpoint_sha256": checkpoint, "fit_solver": "full"}
    save_json(output, info)
    return info


def feature_points(path, pca, window=50, step=15, sequence_column="Sequence", feature="lag", max_lag=30, unknown_policy="error"):
    frame = read_table(path)
    if sequence_column not in frame:
        raise ValueError(f"{path}: missing {sequence_column}")
    if feature == "structure" and "Q3" not in frame:
        raise ValueError(f"{path}: missing Q3")
    vectors, metadata, values = [], [], []
    for row_index, window_index, vector in latent_rows(frame, str(path)):
        row = frame.iloc[row_index]
        sequence = "".join(row[sequence_column].split()).upper()
        starts = sliding_positions(len(sequence), window, step)
        if window_index >= len(starts):
            raise ValueError(f"{path}, data row {row_index + 1}: z_{window_index} has no matching complete sequence window")
        start = starts[window_index]
        if feature == "structure":
            q3 = "".join(row["Q3"].split()).upper()
            if len(q3) != len(sequence) or set(q3) - set("HEC"):
                raise ValueError(f"{path}, data row {row_index + 1}: Q3 must match Sequence length and use H/E/C")
            values.append(q3_fraction(q3[start:start + window]))
        else:
            encoded = encode_sequence(sequence, "protein", unknown_policy)
            hlb = np.array([HLB[m] for m in encoded[start:start + window]])
            values.append(lag_profile(hlb, max_lag))
        vectors.append(vector)
        metadata.append({"source_file": Path(path).name, "source_row": row_index,
                         "Entry": row.get("Entry", str(row_index)), "window_index": window_index,
                         "start": start, "end": start + window})
    if not vectors:
        raise ValueError(f"No aligned sequence windows in {path}")
    coords = project(np.stack(vectors), pca)
    points = pd.DataFrame(metadata)
    points["PC1"], points["PC2"] = coords[:, 0], coords[:, 1]
    return points, np.stack(values)


def bin_members(points, grid):
    ix, iy = grid.indices(points)
    bins = {}
    for i, (x, y) in enumerate(zip(ix, iy)):
        if 0 <= x < grid.nx and 0 <= y < grid.ny:
            bins.setdefault((x, y), []).append(i)
    return bins


def make_map(path, pca_path, output_dir, kind="lag", grid_size=2, min_points=10,
             window=50, step=15, max_lag=30, overwrite=False, unknown_policy="error", sequence_column="Sequence"):
    if min_points < 1:
        raise ValueError("Minimum points must be positive")
    pca = load_pca(pca_path)
    check_embedding_metadata([path], window, step, pca)
    points, values = feature_points(path, pca, window, step, sequence_column, kind, max_lag, unknown_policy)
    coords = points[["PC1", "PC2"]].to_numpy()
    grid = Grid.from_points(coords, grid_size)
    bins = bin_members(coords, grid)
    rows = []
    if kind == "lag":
        entropies = np.array([lag_entropy(v) for v in values])
        variances = np.nanvar(values, axis=1)
        for lag in range(max_lag):
            points[f"lag_{lag + 1}"] = values[:, lag]
        points["lag_variance"], points["lag_entropy"] = variances, entropies
    else:
        for column, index in [("H", 0), ("E", 1), ("C", 2)]:
            points[column] = values[:, index]
    for x, y, center_x, center_y in grid.cells():
        indices = bins.get((x, y), [])
        supported = len(indices) >= min_points
        record = {"grid_x": center_x, "grid_y": center_y, "count": len(indices)}
        if kind == "lag":
            mean_profile = np.mean(values[indices], axis=0) if supported else np.full(max_lag, np.nan)
            finite_entropy = entropies[indices][np.isfinite(entropies[indices])] if supported else []
            variance = float(np.var(mean_profile)) if supported else np.nan
            entropy = float(np.mean(finite_entropy)) if len(finite_entropy) else np.nan
            record.update({"var_mean_after": variance, "var_point_first": entropy,
                           "variance_of_mean_lag_profile": variance, "mean_lag_entropy": entropy,
                           "mean_segment_lag_variance": float(np.mean(variances[indices])) if supported else np.nan})
            record.update({f"lag_{i + 1}": value for i, value in enumerate(mean_profile)})
        else:
            fractions = np.mean(values[indices], axis=0) if supported else np.full(3, np.nan)
            record.update(dict(zip(["mean_H", "mean_E", "mean_C"], fractions)))
            record["structure_tendency"] = float(structure_tendency(fractions))
        rows.append(record)
    output_dir = Path(output_dir)
    map_csv, points_csv = output_dir / f"{kind}_map.csv", output_dir / f"{kind}_points.csv"
    prepare_outputs([map_csv, points_csv, meta_path(map_csv)], overwrite, [path, pca_path])
    points.to_csv(points_csv, index=False)
    result = pd.DataFrame(rows)
    result.to_csv(map_csv, index=False)
    save_json(meta_path(map_csv), {"kind": kind, "grid": grid.to_dict(), "pca_sha256": sha256(pca_path),
                                 "inputs": input_records([path]), "window": window, "step": step,
                                 "min_points": min_points, "max_lag": max_lag if kind == "lag" else None,
                                 "unknown_policy": unknown_policy})
    return result, grid, map_csv


def score_points(coords, score_grid, grid, pc2_limit=None):
    if pc2_limit is not None and (pc2_limit < 0 or not np.isfinite(pc2_limit)):
        raise ValueError("PC2 limit must be nonnegative and finite")
    eligible = np.ones(len(coords), dtype=bool) if pc2_limit is None else np.abs(coords[:, 1]) <= pc2_limit
    ix, iy = grid.indices(coords)
    scores = np.full(len(coords), np.nan)
    for i, (x, y) in enumerate(zip(ix, iy)):
        if eligible[i] and 0 <= x < grid.nx and 0 <= y < grid.ny and np.isfinite(score_grid[y, x]):
            scores[i] = score_grid[y, x]
    valid = np.isfinite(scores)
    count_eligible = int(eligible.sum())
    return scores, eligible, {"mean_structure_score": float(scores[valid].mean()) if valid.any() else np.nan,
                              "coverage_ratio": int(valid.sum()) / count_eligible if count_eligible else np.nan,
                              "N_points": len(coords), "N_region": count_eligible, "N_scored": int(valid.sum())}


def score_rhp(paths, pca_path, map_path, output_dir, overwrite=False, pc2_limit=None, legacy_grid_size=2):
    pca = load_pca(pca_path)
    check_embedding_metadata(paths, pca=pca)
    frame, grid, _ = load_grid_table(map_path, pca_path, legacy_grid_size)
    if "structure_tendency" not in frame:
        raise ValueError("A structure-tendency map is required")
    score_grid = frame.structure_tendency.to_numpy().reshape(grid.ny, grid.nx)
    results, point_frames = [], []
    for path in paths:
        vectors, rows = read_latents(path)
        coords = project(vectors, pca)
        scores, eligible, record = score_points(coords, score_grid, grid, pc2_limit)
        label = Path(path).stem
        record["RHP"] = label
        results.append(record)
        point_frames.append(pd.DataFrame({"RHP": label, "source_row": [r[0] for r in rows],
                                          "window_index": [r[1] for r in rows], "PC1": coords[:, 0],
                                          "PC2": coords[:, 1], "in_region": eligible, "structure_tendency": scores}))
    output_dir = Path(output_dir)
    summary, points = output_dir / "rhp_scores.csv", output_dir / "rhp_score_points.csv"
    prepare_outputs([summary, points, meta_path(summary)], overwrite, list(paths) + [pca_path, map_path])
    result = pd.DataFrame(results)
    result.to_csv(summary, index=False)
    all_points = pd.concat(point_frames, ignore_index=True)
    all_points.to_csv(points, index=False)
    save_json(meta_path(summary), {"pca_sha256": sha256(pca_path), "map_sha256": sha256(map_path),
                                 "inputs": input_records(paths), "pc2_limit": pc2_limit,
                                 "interpretation": "empirical window-weighted mean of finite protein-derived grid scores"})
    return result, all_points
