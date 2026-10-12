"""Shared encoding, CSV schemas, fixed PCA projection, and grid coordinates."""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

AA_GROUPS = {"1": "ACYTG", "2": "SQHNP", "3": "LIFWVM", "4": "EDRK"}
AA_TO_MONOMER = {aa: code for code, residues in AA_GROUPS.items() for aa in residues}
HLB = {"1": 8.45, "2": 11.42, "3": 5.125, "4": 18.5}
LATENT_DIM = 16


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_records(paths):
    return [{"file": str(Path(p)), "sha256": sha256(p)} for p in paths]


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def meta_path(path):
    return Path(str(path) + ".meta.json")


def prepare_outputs(paths, overwrite=False, inputs=()):
    sources = {Path(p).resolve() for p in inputs}
    for path in map(Path, paths):
        if path.resolve() in sources:
            raise ValueError(f"Output would replace an input: {path}")
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {path}; use --overwrite to replace it")
    for path in map(Path, paths):
        path.parent.mkdir(parents=True, exist_ok=True)


def read_table(path, sequence_column="Sequence"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str, keep_default_na=False)
    raise ValueError(f"Expected a CSV or Excel table: {path}")


def encode_sequence(sequence, kind="protein", unknown_policy="error"):
    sequence = "".join(str(sequence).split()).upper()
    if kind == "rhp":
        bad = set(sequence) - set("1234")
        if bad:
            raise ValueError(f"RHP sequence contains unsupported symbols: {sorted(bad)}")
        return sequence
    bad = set(sequence) - set(AA_TO_MONOMER)
    if bad and unknown_policy != "mma":
        raise ValueError(f"Noncanonical amino acids {sorted(bad)}; use --unknown-policy mma for the original fallback")
    return "".join(AA_TO_MONOMER.get(aa, "1") for aa in sequence)


def sliding_positions(length, window=50, step=15):
    if window < 1 or step < 1:
        raise ValueError("Window and step must be positive")
    return list(range(0, length - window + 1, step))


def one_hot_numpy(sequence, max_length=50):
    """Same layout as upstream seq_one_hot_AE: four channels plus terminator."""
    sequence = encode_sequence(sequence, "rhp")
    if len(sequence) > max_length:
        raise ValueError("Sequence exceeds the one-hot input length")
    values = np.zeros((5, max_length + 1), dtype=np.float32)
    values[4, len(sequence):] = 1
    for position, code in enumerate(sequence):
        values[int(code) - 1, position] = 1
    return values


def z_columns(frame):
    return sorted((c for c in frame.columns if re.fullmatch(r"z_\d+", str(c))), key=lambda c: int(c[2:]))


def parse_z(cell):
    if cell is None or str(cell).strip() == "" or (isinstance(cell, float) and np.isnan(cell)):
        return None
    try:
        value = ast.literal_eval(cell) if isinstance(cell, str) else cell
    except (ValueError, SyntaxError) as exc:
        raise ValueError("Malformed latent-vector cell") from exc
    if not isinstance(value, (list, tuple)) or len(value) != LATENT_DIM:
        raise ValueError("A latent vector must contain exactly 16 numbers")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value):
        raise ValueError("Latent-vector values must be numeric")
    vector = np.asarray(value, dtype=float)
    if not np.isfinite(vector).all():
        raise ValueError("Latent vectors must be finite")
    return vector


def latent_rows(frame, source="input"):
    columns = z_columns(frame)
    if not columns:
        raise ValueError(f"No z_0, z_1, ... columns in {source}")
    for index, row in frame.iterrows():
        for column in columns:
            try:
                vector = parse_z(row[column])
            except ValueError as exc:
                raise ValueError(f"{source}, data row {index + 1}, {column}: {exc}") from exc
            if vector is not None:
                yield index, int(column[2:]), vector


def read_latents(path):
    frame = read_table(path)
    rows = list(latent_rows(frame, str(path)))
    if not rows:
        raise ValueError(f"No valid latent vectors in {path}")
    return np.stack([r[2] for r in rows]), rows


def load_pca(path):
    info = json.loads(Path(path).read_text(encoding="utf-8"))
    mean = np.asarray(info.get("mean"), dtype=float)
    components = np.asarray(info.get("components"), dtype=float)
    if mean.shape != (LATENT_DIM,) or components.ndim != 2 or components.shape[1] != LATENT_DIM or len(components) < 2:
        raise ValueError("PCA JSON requires mean[16] and at least two components of length 16")
    if not np.isfinite(mean).all() or not np.isfinite(components).all():
        raise ValueError("PCA parameters must be finite")
    return info


def project(values, pca):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != LATENT_DIM:
        raise ValueError("PCA input must have shape (N, 16)")
    return ((values - np.asarray(pca["mean"])) @ np.asarray(pca["components"]).T)[:, :2]


def check_embedding_metadata(paths, window=None, step=None, pca=None):
    checkpoints = set()
    for path in paths:
        sidecar = meta_path(path)
        if not sidecar.exists():
            continue  # Legacy wide CSV files have no sidecars.
        info = json.loads(sidecar.read_text())
        if info.get("output_csv_sha256") and info["output_csv_sha256"] != sha256(path):
            raise ValueError(f"Embedding sidecar no longer matches {path}")
        for key, expected in [("window", window), ("step", step)]:
            if expected is not None and key in info and info[key] != expected:
                raise ValueError(f"{path}: {key} differs from the embedding settings")
        if info.get("checkpoint_sha256"):
            checkpoints.add(info["checkpoint_sha256"])
    if len(checkpoints) > 1:
        raise ValueError("Inputs were generated by different encoder checkpoints")
    if pca and pca.get("checkpoint_sha256") and checkpoints and checkpoints != {pca["checkpoint_sha256"]}:
        raise ValueError("Encoder checkpoint differs from the PCA reference")
    return next(iter(checkpoints), None)


@dataclass(frozen=True)
class Grid:
    x0: float
    y0: float
    nx: int
    ny: int
    size: float

    @classmethod
    def from_points(cls, points, size=2.0):
        if size <= 0 or not np.isfinite(size):
            raise ValueError("Grid size must be positive and finite")
        x0 = math.floor(float(points[:, 0].min()) / size) * size
        y0 = math.floor(float(points[:, 1].min()) / size) * size
        nx = math.floor((float(points[:, 0].max()) - x0) / size) + 1
        ny = math.floor((float(points[:, 1].max()) - y0) / size) + 1
        if nx * ny > 1_000_000:
            raise ValueError("Grid has over one million cells; increase --grid-size")
        return cls(x0, y0, nx, ny, float(size))

    @property
    def extent(self):
        return [self.x0, self.x0 + self.nx * self.size, self.y0, self.y0 + self.ny * self.size]

    def indices(self, points):
        ix = np.floor((points[:, 0] - self.x0) / self.size).astype(int)
        iy = np.floor((points[:, 1] - self.y0) / self.size).astype(int)
        return ix, iy

    def cells(self):
        return [(x, y, self.x0 + (x + .5) * self.size, self.y0 + (y + .5) * self.size)
                for y in range(self.ny) for x in range(self.nx)]

    def to_dict(self):
        return asdict(self)


def load_grid_table(path, pca_path=None, legacy_grid_size=2.0):
    frame = pd.read_csv(path)
    if not {"grid_x", "grid_y"} <= set(frame):
        raise ValueError("Map CSV requires grid_x and grid_y")
    sidecar = meta_path(path)
    info = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    if pca_path and info.get("pca_sha256") and info["pca_sha256"] != sha256(pca_path):
        raise ValueError("Map and requested PCA reference differ")
    xs = np.sort(frame.grid_x.unique())
    ys = np.sort(frame.grid_y.unique())
    if len(frame) != len(xs) * len(ys) or frame.duplicated(["grid_x", "grid_y"]).any():
        raise ValueError("Map must contain every grid cell exactly once")
    if "grid" in info:
        grid = Grid(**info["grid"])
    else:
        spacings = [np.diff(a) for a in (xs, ys) if len(a) > 1]
        size = float(spacings[0][0]) if spacings else float(legacy_grid_size)
        if size <= 0 or any(not np.allclose(a, size) for a in spacings):
            raise ValueError("Legacy map does not have a uniform square grid")
        grid = Grid(float(xs[0] - size / 2), float(ys[0] - size / 2), len(xs), len(ys), size)
    expected_x = grid.x0 + (np.arange(grid.nx) + .5) * grid.size
    expected_y = grid.y0 + (np.arange(grid.ny) + .5) * grid.size
    if len(xs) != grid.nx or len(ys) != grid.ny or not np.allclose(xs, expected_x) or not np.allclose(ys, expected_y):
        raise ValueError("Map coordinate centers do not match its grid metadata")
    return frame.sort_values(["grid_y", "grid_x"]).reset_index(drop=True), grid, info
