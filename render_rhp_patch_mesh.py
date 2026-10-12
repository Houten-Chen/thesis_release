#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render an RHP as matching continuous and colored triangular SAS surfaces.

Install (Python 3.10+):
    python -m pip install numpy scipy scikit-image matplotlib pillow vtk

Examples:
    python render_rhp_patch_mesh.py seq151_1.pdb
    python render_rhp_patch_mesh.py seq151_1.pdb seq658_3.pdb --out comparison
    python render_rhp_patch_mesh.py seq151_1.pdb --interactive --export-vtp
    python render_rhp_patch_mesh.py "seq*.pdb" --grid-spacing 0.35

Interactive preview: drag either panel to rotate BOTH views; wheel zooms;
Enter exports that view; Escape / closing the window skips the input.
Default mode saves automatically. Linux without a display needs VTK >=9.5
with EGL; --renderer matplotlib is a portable, slower fallback.

The analysis core is copied from the existing rhp_surface_patches.py. It uses
an unsmoothed external SAS, explicit hydrogens, radius-aware vertex ownership,
EHMA C7..C14 plus attached hydrogens, and original mesh-edge connectivity.
All coordinates are Angstroms; areas are Angstrom squared.

To visualize the original one-third-per-vertex area allocation exactly, each
original face is divided into SIX COPLANAR triangles, two per corner. This
changes neither the molecular geometry nor the calculated patch areas.
Dark edges show the ORIGINAL triangular mesh. In enlarged panels, thin pale
lines show the equal-area corner regions. The left and right views use the
same geometry, camera and illumination; only mesh lines/color modes differ.
The monomer comparison colors whole repeat units. The separate EHMA figure
colors only the alkyl selection used for patch counting, then its components.

Outputs per PDB: comparison PNG, EHMA-patch PNG, separate clean panel PNGs,
method JSON, patch CSV and original surface_mesh.npz. --export-vtp also saves
a connected display mesh with exact cell colors, source vertices and patch IDs.
No smoothing, mesh decimation, PBC unwrapping or minimum-patch filter is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree, distance
from skimage.measure import marching_cubes

VERSION = "1.0.0"
MONOMERS = ("MMA", "EHMA", "OEGMA", "SPMA")
# Explicit-atom, Bondi-style VdW radii in Angstrom. Not MD Lennard-Jones sigma.
VDW_RADII = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52,
             "F": 1.47, "P": 1.80, "S": 1.80}
# Used ONLY to check coordinate continuity, not to parameterize the surface.
COVALENT_RADII = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66,
                  "F": 0.57, "P": 1.07, "S": 1.05}
RESIDUE_MAP = {
    "MM1": "MMA", "MM2": "MMA", "HM1": "MMA", "HM2": "MMA", "TMM": "MMA",
    "EH1": "EHMA", "EH2": "EHMA", "HE1": "EHMA", "HE2": "EHMA", "TEH": "EHMA",
    "OE1": "OEGMA", "OE2": "OEGMA", "HO1": "OEGMA", "HO2": "OEGMA", "TOE": "OEGMA",
    "SP1": "SPMA", "SP2": "SPMA", "HS1": "SPMA", "HS2": "SPMA", "TSP": "SPMA",
}
SOLVENT_IONS = {"HOH", "WAT", "SOL", "TIP", "NA", "K", "CL", "CA", "MG"}
BACKBONE_NAMES = {"C1", "C2", "C3", "C4", "C5"}
ESTER_NAMES = {"C6", "O1", "O2"}
COLORS = {"MMA": (166, 206, 227), "EHMA": (242, 117, 49),
          "OEGMA": (100, 190, 154), "SPMA": (171, 117, 192)}


@dataclass
class Atom:
    serial: int
    name: str
    resname: str
    chain: str
    resid: str
    segment: int
    element: str
    xyz: tuple[float, float, float]
    monomer: str
    occupancy: float = 1.0
    parent: int = -1  # array index of this heavy atom or a hydrogen's parent
    region: str = ""
    ehma_alkyl: bool = False

    @property
    def residue_key(self):
        return self.segment, self.chain, self.resid

    @property
    def residue_label(self):
        return f"seg{self.segment}:{self.chain or '_'}:{self.resid}:{self.resname}"


def read_json_object(path: Path | None) -> dict:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object.")
    return value


def read_pdb(path: Path, residue_map: dict, model_number: int):
    """Read one MODEL, preserve TER boundaries, keep blank/A altlocs.

    MODEL selection is 1-based by occurrence, not by its printed identifier.
    Solvent/ions are ignored. Unknown non-solvent residues fail explicitly.
    """
    records = {}
    warnings = []
    model_index = 1
    seen_model = False
    segment = 0
    ignored = set()
    with path.open(encoding="utf-8", errors="replace") as handle:
        for lineno, line in enumerate(handle, 1):
            record = line[:6].strip()
            if record == "MODEL":
                model_index = model_index + 1 if seen_model else 1
                seen_model = True
                segment = 0
                continue
            if record == "ENDMDL":
                if model_index == model_number:
                    break
                continue
            if model_index != model_number:
                continue
            if record == "TER":
                segment += 1
                continue
            if record not in {"ATOM", "HETATM"}:
                continue
            resname = line[17:20].strip().upper()
            if resname in SOLVENT_IONS:
                ignored.add(resname)
                continue
            if resname not in residue_map:
                raise ValueError(f"Unknown residue {resname!r} at line {lineno}; "
                                 "provide --residue-map with an explicit assignment.")
            alt = line[16:17].strip()
            if alt not in {"", "A"}:
                continue
            try:
                name = line[12:16].strip()
                element = line[76:78].strip().upper()
                if not element:
                    element = re.sub(r"^[0-9]+", "", name)[0].upper()
                atom = Atom(int(line[6:11]), name, resname, line[21:22].strip(),
                            line[22:27].strip(), segment, element,
                            tuple(float(line[i:i + 8]) for i in (30, 38, 46)),
                            residue_map[resname],
                            float(line[54:60].strip() or "1"))
            except (ValueError, IndexError) as exc:
                raise ValueError(f"Malformed ATOM/HETATM at line {lineno}.") from exc
            key = atom.residue_key + (atom.name,)
            if key in records:
                # A duplicate without an altloc is usually malformed input.
                old, old_alt = records[key]
                if not old_alt and not alt:
                    raise ValueError(f"Duplicate atom name {atom.name} in {atom.residue_label}.")
                if atom.occupancy <= old.occupancy:
                    continue
            records[key] = atom, alt
    atoms = [value[0] for value in records.values()]
    if not atoms:
        raise ValueError(f"No polymer atoms in MODEL occurrence {model_number}.")
    if ignored:
        warnings.append("Ignored solvent/ions: " + ", ".join(sorted(ignored)))
    if not any(a.element == "H" for a in atoms):
        warnings.append("No explicit hydrogens found. These radii are not united-atom "
                        "radii; add hydrogens before comparing with the supplied examples.")
    return atoms, warnings


def classify_and_check(atoms, carbon_names):
    """Validate intact coordinates and assign hydrogens via nearest heavy atom.

    No PBC wrapping or bond reconstruction is attempted. Fail on disconnected
    repeat units or long C3(previous)-C2(next) backbone bonds. This check assumes
    the naming convention used in the supplied PDBs.
    """
    residues = {}
    for i, atom in enumerate(atoms):
        residues.setdefault(atom.residue_key, []).append(i)
        if atom.element not in COVALENT_RADII:
            raise ValueError(f"Unsupported element {atom.element} at atom {atom.serial}.")
    last_by_chain = {}
    max_backbone_bond = 0.0
    for key, ids in residues.items():
        local = [atoms[i] for i in ids]
        if len({a.resname for a in local}) != 1:
            raise ValueError(f"Mixed residue names for residue {key}.")
        xyz = np.array([a.xyz for a in local])
        d = distance.cdist(xyz, xyz)
        cov = np.array([COVALENT_RADII[a.element] for a in local])
        adjacency = (d > 0.35) & (d < cov[:, None] + cov[None, :] + 0.25)
        n_components, _ = connected_components(coo_matrix(adjacency), directed=False)
        if n_components != 1:
            raise ValueError(f"{local[0].residue_label} has {n_components} disconnected "
                             "parts under a covalent-distance check. Make the molecule "
                             "whole across PBC with the MD topology before analysis.")
        named = {a.name: i for i, a in zip(ids, local)}
        if "C2" not in named or "C3" not in named:
            raise ValueError(f"{local[0].residue_label}: missing backbone C2 or C3.")
        chain_key = key[:2]
        if chain_key in last_by_chain:
            length = np.linalg.norm(np.array(atoms[named["C2"]].xyz)
                                    - np.array(atoms[last_by_chain[chain_key]].xyz))
            max_backbone_bond = max(max_backbone_bond, float(length))
            if not 0.8 < length < 2.2:
                raise ValueError(f"Backbone bond before {local[0].residue_label} is "
                                 f"{length:.2f} A; coordinates may be split by PBC, "
                                 "or chains need distinct chain IDs / TER records.")
        last_by_chain[chain_key] = named["C3"]
        if local[0].monomer == "EHMA":
            found = {a.name for a in local if a.element == "C"}
            missing = set(carbon_names) - found
            if missing:
                raise ValueError(f"{local[0].residue_label}: missing EHMA alkyl "
                                 f"carbons {sorted(missing)}; adjust --ehma-carbons.")
        heavy = [i for i in ids if atoms[i].element != "H"]
        heavy_xyz = np.array([atoms[i].xyz for i in heavy])
        for i in ids:
            atom = atoms[i]
            if atom.element == "H":
                hd = np.linalg.norm(heavy_xyz - atom.xyz, axis=1)
                nearest = int(np.argmin(hd))
                if not 0.5 < hd[nearest] < 1.5:
                    raise ValueError(f"Cannot assign hydrogen {atom.serial} to a "
                                     f"heavy atom in {atom.residue_label}.")
                atom.parent = heavy[nearest]
            else:
                atom.parent = i
            parent_name = atoms[atom.parent].name
            atom.region = ("backbone_common" if parent_name in BACKBONE_NAMES else
                           "ester" if parent_name in ESTER_NAMES else "sidechain")
            atom.ehma_alkyl = (atom.monomer == "EHMA"
                              and atoms[atom.parent].element == "C"
                              and parent_name in carbon_names)
    return residues, max_backbone_bond


def make_sas(xyz, expanded_radii, spacing, max_cells, include_cavities):
    """Marching-cubes boundary of min_i(|x-c_i|-expanded_radius_i) = 0.

    Atom-local rasterization avoids an all-grid-points x all-atoms matrix.
    Optional cavity removal keeps only surfaces reachable by the probe center
    from the exterior of the finite grid (six-neighbor voxel connectivity).
    """
    origin = np.min(xyz - expanded_radii[:, None], axis=0) - 3 * spacing
    upper = np.max(xyz + expanded_radii[:, None], axis=0) + 3 * spacing
    shape = np.ceil((upper - origin) / spacing).astype(int) + 1
    n_cells = int(np.prod(shape.astype(np.int64)))
    if n_cells > max_cells:
        raise ValueError(f"Grid {tuple(shape)} needs {n_cells:,} cells, above "
                         f"--max-grid-cells {max_cells:,}. Check coordinates or "
                         "increase --grid-spacing / --max-grid-cells.")
    field = np.full(tuple(shape), 3 * spacing, dtype=np.float32)
    for center, radius in zip(xyz, expanded_radii):
        extent = radius + 2 * spacing
        lo = np.maximum(0, np.floor((center - extent - origin) / spacing).astype(int))
        hi = np.minimum(shape, np.ceil((center + extent - origin) / spacing).astype(int) + 1)
        axes = [(origin[k] + spacing * np.arange(lo[k], hi[k]) - center[k]) ** 2
                for k in range(3)]
        signed = np.sqrt(axes[0][:, None, None] + axes[1][None, :, None]
                         + axes[2][None, None, :]) - radius
        block = tuple(slice(int(lo[k]), int(hi[k])) for k in range(3))
        np.minimum(field[block], signed, out=field[block])
    removed_cavity_cells = 0
    if not include_cavities:
        inside = field <= 0
        filled = ndimage.binary_fill_holes(inside)
        hidden_void = filled & ~inside
        removed_cavity_cells = int(np.count_nonzero(hidden_void))
        field[hidden_void] = -spacing
        del inside, filled, hidden_void
    vertices, faces, _, _ = marching_cubes(
        field, level=0.0, spacing=(spacing, spacing, spacing),
        gradient_direction="ascent", allow_degenerate=False)
    vertices = vertices.astype(np.float64) + origin
    del field
    tri = vertices[faces]
    face_area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0],
                                        tri[:, 2] - tri[:, 0]), axis=1) / 2.0
    vertex_area = np.bincount(faces.ravel(), weights=np.repeat(face_area / 3.0, 3),
                              minlength=len(vertices))
    return vertices, faces, vertex_area, {
        "grid_shape": shape.tolist(), "grid_cells": n_cells,
        "grid_origin_A": origin.tolist(), "vertices": len(vertices),
        "triangles": len(faces), "removed_cavity_grid_cells": removed_cavity_cells,
    }


def assign_owner(vertices, xyz, expanded_radii):
    """Radius-aware ownership: minimize |vertex-center| - expanded_radius.

    One nearest-neighbor query per unique radius is exact for this criterion.
    Nearest atom center alone can assign surfaces to the wrong element radius.
    """
    owner = np.full(len(vertices), -1, dtype=np.int64)
    best = np.full(len(vertices), np.inf)
    for radius in np.unique(expanded_radii):
        ids = np.flatnonzero(expanded_radii == radius)
        dist, local = cKDTree(xyz[ids]).query(vertices, k=1)
        value = dist - radius
        update = value < best
        best[update] = value[update]
        owner[update] = ids[local[update]]
    if np.any(owner < 0):
        raise RuntimeError("Unassigned surface vertices.")
    return owner, float(np.max(np.abs(best)))


def find_ehma_patches(vertices, faces, vertex_area, owner, atoms):
    """Connected selected vertices along actual mesh edges; IDs are size ranked.

    A triangle at a chemical boundary contributes 1/3 or 2/3 of its area to
    EHMA according to its vertex labels. Areas partition the entire mesh.
    """
    selected = np.array([a.ehma_alkyl for a in atoms], dtype=bool)[owner]
    ids = np.flatnonzero(selected)
    patch_id = np.full(len(vertices), -1, dtype=np.int32)
    if len(ids) == 0:
        return patch_id, []
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = edges[selected[edges[:, 0]] & selected[edges[:, 1]]]
    local = np.full(len(vertices), -1, dtype=np.int64)
    local[ids] = np.arange(len(ids))
    graph = coo_matrix((np.ones(len(edges), dtype=np.int8),
                        (local[edges[:, 0]], local[edges[:, 1]])),
                       shape=(len(ids), len(ids))).tocsr()
    graph.data[:] = 1  # only connectivity matters, including duplicate edges
    n_patches, labels = connected_components(graph, directed=False)
    areas = np.bincount(labels, weights=vertex_area[ids], minlength=n_patches)
    ranked = np.argsort(-areas, kind="stable")
    rank = np.empty(n_patches, dtype=np.int32)
    rank[ranked] = np.arange(1, n_patches + 1)
    patch_id[ids] = rank[labels]
    rows = []
    for old in ranked:
        vid = ids[labels == old]
        contributing_atoms = np.unique(owner[vid])
        residue_labels = sorted({atoms[i].residue_label for i in contributing_atoms})
        center = np.average(vertices[vid], weights=vertex_area[vid], axis=0)
        rows.append({"patch_id": int(rank[old]), "area_A2": float(areas[old]),
                     "area_nm2": float(areas[old] / 100.0),
                     "n_vertices": len(vid), "n_contributing_atoms": len(contributing_atoms),
                     "n_EHMA_units": len(residue_labels),
                     "EHMA_residues": ";".join(residue_labels),
                     "center_x_A": float(center[0]), "center_y_A": float(center[1]),
                     "center_z_A": float(center[2])})
    return patch_id, rows


CORE_SOURCE_SHA256 = "6cb38fc2263d3bb955608dfadd9cd9004c65d87d8c4ef91508c55772daa9ff30"

SCRIPT_VERSION = "1.0.0"
COLOR_HEX = {"EHMA": "#E53935", "OEGMA": "#A6CEE3",
             "SPMA": "#174A8B", "MMA": "#A6A6A6"}


def hex_rgb(value):
    return np.array([int(value[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.uint8)


def triangle_areas(vertices, faces):
    tri = vertices[faces]
    return np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0],
                                   tri[:, 2] - tri[:, 0]), axis=1) * 0.5


def barycentric_display_mesh(vertices, faces):
    """Split each triangle into six triangles without changing its surface.

    Each original corner owns two subtriangles, exactly one third of the
    original area. Shared edge midpoints are shared vertices, so the display
    mesh is continuous. Patch connectivity is ALWAYS computed on the original
    mesh, never on this display subdivision.
    """
    f = np.asarray(faces, dtype=np.int64)
    raw_edges = np.stack((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]), axis=1)
    edges, inverse = np.unique(np.sort(raw_edges.reshape(-1, 2), axis=1),
                                axis=0, return_inverse=True)
    mid = inverse.reshape(-1, 3) + len(vertices)
    cent = np.arange(len(f)) + len(vertices) + len(edges)
    display_vertices = np.vstack((vertices, vertices[edges].mean(axis=1),
                                   vertices[f].mean(axis=1)))
    sf = np.stack((
        np.column_stack((f[:, 0], mid[:, 0], cent)),
        np.column_stack((f[:, 0], cent, mid[:, 2])),
        np.column_stack((f[:, 1], mid[:, 1], cent)),
        np.column_stack((f[:, 1], cent, mid[:, 0])),
        np.column_stack((f[:, 2], mid[:, 2], cent)),
        np.column_stack((f[:, 2], cent, mid[:, 1])),
    ), axis=1).reshape(-1, 3)
    source_corner = np.repeat(f, 2, axis=1).reshape(-1)
    # These three segments bound the three equal-area corner regions.
    allocation_lines = np.stack((np.repeat(cent, 3), mid.reshape(-1)), axis=1)
    return display_vertices, sf, source_corner, edges, allocation_lines


def compute_surface(pdb, args):
    residue_map = RESIDUE_MAP | {str(k).upper(): str(v).upper()
                                 for k, v in read_json_object(args.residue_map).items()}
    radii = VDW_RADII | {str(k).upper(): float(v)
                         for k, v in read_json_object(args.radii).items()}
    if any(v not in MONOMERS for v in residue_map.values()):
        raise ValueError("Residue-map values must be MMA, EHMA, OEGMA or SPMA")
    if any(not np.isfinite(v) or v <= 0 for v in radii.values()):
        raise ValueError("Atomic radii must be positive and finite")
    carbon_names = {n.strip() for n in args.ehma_carbons.split(",") if n.strip()}
    if not carbon_names:
        raise ValueError("--ehma-carbons cannot be empty")
    atoms, warnings = read_pdb(pdb, residue_map, args.model)
    residues, max_bond = classify_and_check(atoms, carbon_names)
    xyz = np.array([a.xyz for a in atoms], dtype=float)
    if not np.isfinite(xyz).all():
        raise ValueError("Non-finite PDB coordinates")
    expanded = np.array([radii[a.element] for a in atoms]) + args.probe_radius
    v, f, va, grid = make_sas(xyz, expanded, args.grid_spacing,
                              args.max_grid_cells, args.include_cavities)
    owner, error = assign_owner(v, xyz, expanded)
    pid, patches = find_ehma_patches(v, f, va, owner, atoms)
    monomer = np.array([MONOMERS.index(a.monomer) for a in atoms])[owner]
    dv, df, corner, edges, allocation = barycentric_display_mesh(v, f)
    da = triangle_areas(dv, df)
    face_area = triangle_areas(v, f)
    # Check the entire allocation, including chemical boundaries and patches.
    reconstructed = np.bincount(corner, weights=da, minlength=len(v))
    if not np.allclose(reconstructed, va, rtol=1e-9, atol=1e-9):
        raise RuntimeError("Display subdivision does not reproduce vertex areas")
    if not np.allclose(da.reshape(-1, 6).sum(axis=1), face_area,
                       rtol=1e-9, atol=1e-9):
        raise RuntimeError("Display subdivision does not conserve triangle areas")
    selected = pid[corner] > 0
    display_patch_area = np.bincount(pid[corner][selected], weights=da[selected],
                                    minlength=len(patches) + 1)
    expected = np.array([p["area_A2"] for p in patches])
    if not np.allclose(display_patch_area[1:], expected, rtol=1e-9, atol=1e-7):
        raise RuntimeError("Colored patch areas differ from the analysis")
    areas = expected
    summary = {
        "script_version": SCRIPT_VERSION, "pdb": pdb.name,
        "core_source_sha256": CORE_SOURCE_SHA256,
        "surface_definition": "unsmoothed union of vdW+probe spheres; " +
            ("enclosed cavity surfaces included" if args.include_cavities else "external SAS only"),
        "grid_spacing_A": args.grid_spacing, "probe_radius_A": args.probe_radius,
        "vdw_radii_A": radii, "include_cavities": args.include_cavities,
        "coordinate_units": "Angstrom", "area_units": "Angstrom^2",
        "model_occurrence": args.model, "n_atoms": len(atoms),
        "n_repeat_units": len(residues), "max_backbone_bond_A": max_bond,
        "vertices": len(v), "triangles": len(f),
        "display_vertices": len(dv), "display_triangles": len(df),
        "total_surface_area_A2": float(va.sum()),
        "EHMA_alkyl_area_A2": float(areas.sum()), "EHMA_patch_count": len(patches),
        "EHMA_area_weighted_mean_A2": float(areas @ areas / areas.sum())
            if len(areas) and areas.sum() else None,
        "largest_EHMA_patch_A2": float(areas.max()) if len(areas) else 0.0,
        "EHMA_alkyl_carbons": sorted(carbon_names),
        "EHMA_hydrogens": "included when attached to the selected alkyl carbons",
        "vertex_ownership": "argmin(distance - vdW_radius - probe_radius)",
        "patch_connectivity": "EHMA alkyl vertices connected along original mesh edges",
        "area_allocation": "one third of each original triangle to each corner vertex",
        "display_allocation": "six coplanar subtriangles; two per original corner",
        "geometric_smoothing": False, "mesh_decimation": False,
        "subdivision_area_conservation_verified": True,
        "max_vertex_distance_from_exact_SAS_A": error, "warnings": warnings,
        "monomer_colors": COLOR_HEX, **grid,
    }
    print(f"{pdb.name}: {len(v):,} vertices, {len(f):,} original triangles; "
          f"{len(patches)} EHMA alkyl patches; area allocation verified", flush=True)
    return {"pdb": pdb, "atoms": atoms, "v": v, "f": f, "va": va,
            "owner": owner, "pid": pid, "monomer": monomer, "patches": patches,
            "dv": dv, "df": df, "corner": corner, "edges": edges,
            "allocation": allocation, "summary": summary}


def patch_palette(n):
    import colorsys
    colors = np.zeros((n + 1, 3), dtype=np.uint8)
    colors[0] = hex_rgb("#DFE2E6")
    for i in range(1, n + 1):
        colors[i] = np.round(np.array(colorsys.hsv_to_rgb(
            ((i - 1) * 0.61803398875) % 1, 0.64, 0.87)) * 255)
    if n:
        colors[1] = hex_rgb(COLOR_HEX["EHMA"])
    return colors


def display_colors(surface, mode):
    corner = surface["corner"]
    if mode == "monomer":
        palette = np.array([hex_rgb(COLOR_HEX[m]) for m in MONOMERS])
        return palette[surface["monomer"][corner]]
    pid = surface["pid"][corner]
    if mode == "patches":
        return patch_palette(len(surface["patches"]))[np.maximum(pid, 0)]
    colors = np.tile(hex_rgb("#DFE2E6"), (len(corner), 1))
    colors[pid > 0] = hex_rgb(COLOR_HEX["EHMA"])
    return colors


def initial_camera(surface, args):
    elev, azim = np.radians([args.elev, args.azim])
    outward = np.array([np.cos(elev) * np.cos(azim),
                        np.cos(elev) * np.sin(azim), np.sin(elev)])
    right = np.array([-np.sin(azim), np.cos(azim), 0.0])
    up = np.cross(outward, right)
    v = surface["v"]
    center = (v.max(axis=0) + v.min(axis=0)) * 0.5
    span = max(np.ptp(v @ right), np.ptp(v @ up))
    return {"position": (center + outward * np.linalg.norm(np.ptp(v, axis=0)) * 4).tolist(),
            "focal_point": center.tolist(), "view_up": up.tolist(),
            "parallel_scale": float(span * 0.56)}


def camera_basis(camera):
    focus = np.array(camera["focal_point"])
    outward = np.array(camera["position"]) - focus
    outward /= np.linalg.norm(outward)
    up = np.array(camera["view_up"])
    right = np.cross(up, outward)
    right /= np.linalg.norm(right)
    up = np.cross(outward, right)
    return focus, outward, right, up


def choose_detail(surface, camera, args, label_kind="monomer"):
    if args.detail_center is not None:
        center = np.array(args.detail_center, dtype=float)
    else:
        focus, outward, right, up = camera_basis(camera)
        v, f = surface["v"], surface["f"]
        centers = v[f].mean(axis=1)
        projection = np.column_stack(((centers - focus) @ right,
                                      (centers - focus) @ up))
        depth = centers @ outward
        # A front-most depth test prevents picking a buried chemical boundary.
        bins = np.floor(projection / 0.8).astype(int)
        _, inverse = np.unique(bins, axis=0, return_inverse=True)
        front = np.full(inverse.max() + 1, -np.inf)
        np.maximum.at(front, inverse, depth)
        if label_kind == "alkyl":
            label = surface["pid"][f] > 0
            mixed = np.any(label != label[:, :1], axis=1)
            candidate = mixed & np.any(label, axis=1)
        else:
            label = surface["monomer"][f]
            mixed = np.any(label != label[:, :1], axis=1)
            ehma = MONOMERS.index("EHMA")
            candidate = mixed & np.any(label == ehma, axis=1)
        candidate &= depth >= front[inverse] - 0.3
        ids = np.flatnonzero(candidate)
        if not len(ids):
            ids = np.flatnonzero(depth >= front[inverse] - 0.3)
        center = centers[ids[np.argmin(np.linalg.norm(projection[ids], axis=1))]]
    focus, outward, right, up = camera_basis(camera)
    distance = np.linalg.norm(np.array(camera["position"]) - focus)
    detail = {"position": (center + outward * distance).tolist(),
              "focal_point": center.tolist(), "view_up": up.tolist(),
              "parallel_scale": args.detail_span * 0.5}
    return detail


def vtk_polydata(vertices, faces):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    mesh = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(vertices), deep=True))
    mesh.SetPoints(points)
    cells = vtk.vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(len(faces) + 1, dtype=np.int64) * 3, deep=True),
                  numpy_to_vtkIdTypeArray(np.ascontiguousarray(faces, dtype=np.int64).ravel(), deep=True))
    mesh.SetPolys(cells)
    return mesh


def vtk_line_actor(points, lines, color, width, opacity=1.0):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    mesh = vtk.vtkPolyData()
    mesh.SetPoints(points)
    cells = vtk.vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(len(lines) + 1, dtype=np.int64) * 2, deep=True),
                  numpy_to_vtkIdTypeArray(np.ascontiguousarray(lines, dtype=np.int64).ravel(), deep=True))
    mesh.SetLines(cells)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)
    mapper.SetResolveCoincidentTopologyToPolygonOffset()
    mapper.SetRelativeCoincidentTopologyLineOffsetParameters(-1.0, -1.0)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(*color)
    prop.SetLineWidth(width)
    prop.SetOpacity(opacity)
    prop.LightingOff()
    return actor


def vtk_scene(surface, modes, camera, size, wire, allocation, interactive=False):
    import os
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
    mesh = vtk_polydata(surface["dv"], surface["df"])
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(mesh)
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.Update()
    geometry = normals.GetOutput()
    if not interactive and not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        if hasattr(vtk, "vtkEGLRenderWindow"):
            window = vtk.vtkEGLRenderWindow()
        else:
            raise RuntimeError("Headless VTK needs VTK >=9.5; install a current VTK or use --renderer matplotlib")
    else:
        window = vtk.vtkRenderWindow()
    window.SetSize(*size)
    window.SetMultiSamples(8)
    if not interactive:
        window.SetOffScreenRendering(1)
    shared = vtk.vtkCamera()
    shared.SetPosition(*camera["position"])
    shared.SetFocalPoint(*camera["focal_point"])
    shared.SetViewUp(*camera["view_up"])
    shared.SetParallelScale(camera["parallel_scale"])
    shared.ParallelProjectionOn()
    for i, mode in enumerate(modes):
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(1, 1, 1)
        renderer.SetViewport(i / len(modes), 0, (i + 1) / len(modes), 1)
        renderer.SetActiveCamera(shared)
        colored = vtk.vtkPolyData()
        colored.ShallowCopy(geometry)
        colors = numpy_to_vtk(display_colors(surface, mode), deep=True,
                               array_type=vtk.VTK_UNSIGNED_CHAR)
        colors.SetName("RGB")
        colored.GetCellData().SetScalars(colors)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(colored)
        mapper.SetScalarModeToUseCellData()
        mapper.SetColorModeToDirectScalars()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetInterpolationToPhong()
        prop.SetAmbient(0.36)
        prop.SetDiffuse(0.64)
        prop.SetSpecular(0.07)
        prop.SetSpecularPower(24)
        renderer.AddActor(actor)
        if wire[i]:
            renderer.AddActor(vtk_line_actor(geometry.GetPoints(), surface["edges"],
                                (0.14, 0.17, 0.20), 0.9 if allocation else 0.5,
                                0.78 if allocation else 0.40))
        if allocation and wire[i]:
            renderer.AddActor(vtk_line_actor(geometry.GetPoints(), surface["allocation"],
                                (0.96, 0.97, 0.98), 0.75, 0.85))
        window.AddRenderer(renderer)
        renderer.ResetCameraClippingRange()
    return window, shared


def select_camera(surface, camera, args):
    import vtk
    window, shared = vtk_scene(surface, ["monomer", "monomer"], camera,
                                (1400, 700), [False, True], False, interactive=True)
    window.SetWindowName("RHP continuous / triangular surface | Drag: rotate | Enter: export | Esc: skip")
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    accepted = False
    def key(obj, _):
        nonlocal accepted
        if obj.GetKeySym() in {"Return", "KP_Enter"}:
            accepted = True
            obj.TerminateApp()
        elif obj.GetKeySym() == "Escape":
            obj.TerminateApp()
    interactor.AddObserver("KeyPressEvent", key, 1.0)
    interactor.Initialize()
    window.Render()
    interactor.Start()
    chosen = {"position": list(shared.GetPosition()),
              "focal_point": list(shared.GetFocalPoint()),
              "view_up": list(shared.GetViewUp()),
              "parallel_scale": shared.GetParallelScale()}
    window.Finalize()
    return chosen if accepted else None


def capture_vtk(surface, modes, camera, size, wire, allocation):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    window, _ = vtk_scene(surface, modes, camera, size, wire, allocation)
    try:
        window.Render()
        capture = vtk.vtkWindowToImageFilter()
        capture.SetInput(window)
        capture.SetInputBufferTypeToRGB()
        capture.ReadFrontBufferOff()
        capture.Update()
        im = capture.GetOutput()
        w, h, _ = im.GetDimensions()
        return vtk_to_numpy(im.GetPointData().GetScalars()).reshape(h, w, 3)[::-1].copy()
    finally:
        window.Finalize()


def capture_matplotlib(surface, modes, camera, size, wire, allocation):
    """Portable fallback; the geometry and corner area colors remain identical."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    focus, outward, right, up = camera_basis(camera)
    # Transform to the exact saved orthographic camera basis.
    v = np.column_stack(((surface["dv"] - focus) @ right,
                         (surface["dv"] - focus) @ up,
                         (surface["dv"] - focus) @ outward))
    tri = v[surface["df"]]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    light = np.array([-0.35, 0.45, 0.82])
    light /= np.linalg.norm(light)
    shade = 0.60 + 0.40 * np.maximum(n @ light, 0)
    fig = plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=100, facecolor="white")
    half = camera["parallel_scale"]
    aspect = size[0] / len(modes) / size[1]
    for i, mode in enumerate(modes):
        ax = fig.add_axes([i / len(modes), 0, 1 / len(modes), 1], projection="3d")
        ax.add_collection3d(Poly3DCollection(tri,
            facecolors=display_colors(surface, mode) / 255 * shade[:, None],
            edgecolors="none", linewidths=0, antialiased=False))
        if wire[i]:
            ax.add_collection3d(Line3DCollection(v[surface["edges"]],
                colors=(0.14, 0.17, 0.20, 0.7 if allocation else 0.35),
                linewidths=0.55 if allocation else 0.22))
        if allocation and wire[i]:
            ax.add_collection3d(Line3DCollection(v[surface["allocation"]],
                colors=(0.96, 0.97, 0.98, 0.7), linewidths=0.35))
        depth = max(2 * half, float(np.ptp(v[:, 2])))
        ax.set_xlim(-half * aspect, half * aspect)
        ax.set_ylim(-half, half)
        ax.set_zlim(-depth / 2, depth / 2)
        ax.set_box_aspect((2 * half * aspect, 2 * half, depth), zoom=1.45)
        ax.set_proj_type("ortho")
        ax.view_init(elev=90, azim=-90)
        ax.set_axis_off()
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return image


def compose_comparison(surface, camera, detail, args, kind):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Arial"
    from matplotlib.patches import Patch, Rectangle
    from PIL import Image
    capture = capture_vtk if args.renderer == "vtk" else capture_matplotlib
    modes = ["monomer", "monomer"] if kind == "comparison" else ["alkyl", "patches"]
    titles = ["a  Continuous surface", "b  Colored triangular mesh"] if kind == "comparison" else [
        "a  EHMA alkyl surface", "b  Connected EHMA alkyl patches"]
    print(f"  Rendering {kind}: whole surface and matching local detail", flush=True)
    full = capture(surface, modes, camera, (2 * args.pixels, args.pixels), [False, True], False)
    zoom = capture(surface, modes, detail, (2 * args.pixels, int(args.pixels * 0.6)),
                   [False, True], True)
    stem = surface["pdb"].stem
    base = args.out / (stem + ("_comparison" if kind == "comparison" else "_EHMA_patches"))
    Image.fromarray(full[:, :args.pixels]).save(str(base) + "_surface.png")
    Image.fromarray(full[:, args.pixels:]).save(str(base) + "_mesh.png")
    fig = plt.figure(figsize=(10.4, 8.5), dpi=args.dpi, facecolor="white")
    fig.text(0.96, 0.972, f"SAS grid {args.grid_spacing:g} Å", ha="right", fontsize=11, color="#555555")
    full_axes = []
    for i in range(2):
        x = 0.025 + i * 0.50
        fig.text(x + 0.018, 0.928, titles[i], fontsize=15, weight="bold", color="#222222")
        ax = fig.add_axes([x, 0.365, 0.45, 0.55])
        ax.imshow(full[:, i * args.pixels:(i + 1) * args.pixels])
        ax.set_axis_off()
        full_axes.append(ax)
        dz = fig.add_axes([x, 0.105, 0.45, 0.25])
        dz.imshow(zoom[:, i * args.pixels:(i + 1) * args.pixels])
        dz.set_xticks([])
        dz.set_yticks([])
        for spine in dz.spines.values():
            spine.set_color("#B7BEC6")
            spine.set_linewidth(0.65)
    # Mark the very same physical region on both full views.
    focus, outward, right, up = camera_basis(camera)
    delta = np.array(detail["focal_point"]) - focus
    scale = camera["parallel_scale"]
    cx = (0.5 + delta @ right / (2 * scale)) * args.pixels
    cy = (0.5 - delta @ up / (2 * scale)) * args.pixels
    w = args.detail_span * (1 / 0.6) / (2 * scale) * args.pixels
    h = args.detail_span / (2 * scale) * args.pixels
    for ax in full_axes:
        ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h,
                                fill=False, edgecolor="#414850", linewidth=0.75))
    fig.text(0.5, 0.367, "Same local region enlarged", fontsize=12,
             color="#555555", ha="center")
    if kind == "comparison":
        handles = [Patch(facecolor=COLOR_HEX[m], label=m) for m in ("EHMA", "OEGMA", "SPMA", "MMA")]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.061),
                   ncol=4, frameon=False, fontsize=10)
        footer = "Each vertex receives one third of its adjoining triangle areas."
    else:
        s = surface["summary"]
        fig.text(0.5, 0.069,
            f"{s['EHMA_patch_count']} patches  ·  total alkyl area {s['EHMA_alkyl_area_A2']:.1f} Å²"
            f"  ·  largest patch {s['largest_EHMA_patch_A2']:.1f} Å²",
            ha="center", fontsize=12, color="#333333")
        footer = "Patch area is the sum of its connected vertices' area contributions."
    fig.text(0.5, 0.026, footer, fontsize=12, ha="center", color="#555555")
    # Finish and validate a PNG before exposing its final filename.
    import os
    target = Path(str(base) + ".png")
    temporary = target.with_name(target.stem + ".tmp.png")
    fig.savefig(temporary, dpi=args.dpi, facecolor="white")
    with Image.open(temporary) as check:
        check.verify()
    os.replace(temporary, target)
    plt.close(fig)
    return str(base) + ".png"


def guess_rhp(stem):
    match = re.match(r"seq(15|30|40|50|65)", stem, re.IGNORECASE)
    return ("RHP" + str(("15", "30", "40", "50", "65").index(match[1]) + 1)) if match else "RHP"


def save_analysis(surface, camera, detail, args, outputs, patch_detail=None):
    base = args.out / surface["pdb"].stem
    summary = surface["summary"] | {"camera": camera, "detail_camera": detail,
                "patch_detail_camera": patch_detail,
                "renderer": args.renderer, "outputs": [Path(p).name for p in outputs]}
    Path(str(base) + "_method.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    fields = ["patch_id", "area_A2", "area_nm2", "n_vertices", "n_contributing_atoms",
              "n_EHMA_units", "EHMA_residues", "center_x_A", "center_y_A", "center_z_A"]
    with Path(str(base) + "_EHMA_patches.csv").open("w", newline="", encoding="utf-8-sig") as h:
        writer = csv.DictWriter(h, fieldnames=fields)
        writer.writeheader()
        writer.writerows(surface["patches"])
    np.savez_compressed(str(base) + "_surface_mesh.npz", vertices_A=surface["v"],
        faces=surface["f"], vertex_area_A2=surface["va"], owner_atom_index=surface["owner"],
        EHMA_patch_id=surface["pid"], vertex_monomer_id=surface["monomer"],
        monomer_names=np.array(MONOMERS),
        atom_serial=np.array([a.serial for a in surface["atoms"]]))
    if args.export_vtp:
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk
        mesh = vtk_polydata(surface["dv"], surface["df"])
        for name, data in (("source_vertex", surface["corner"]),
                           ("monomer_id", surface["monomer"][surface["corner"]]),
                           ("EHMA_patch_id", surface["pid"][surface["corner"]])):
            arr = numpy_to_vtk(data, deep=True)
            arr.SetName(name)
            mesh.GetCellData().AddArray(arr)
        arr = numpy_to_vtk(display_colors(surface, "monomer"), deep=True,
                           array_type=vtk.VTK_UNSIGNED_CHAR)
        arr.SetName("monomer_RGB")
        mesh.GetCellData().SetScalars(arr)
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(base) + "_colored_surface.vtp")
        writer.SetInputData(mesh)
        if writer.Write() != 1:
            raise OSError("Could not write the VTP display mesh")


def main():
    import glob
    import os
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdb", nargs="+", help="One or more PDB paths; quoted wildcards also work")
    parser.add_argument("--out", type=Path, default=Path("rhp_patch_mesh_output"))
    parser.add_argument("--grid-spacing", "--spacing", type=float, default=0.35)
    parser.add_argument("--probe-radius", "--probe", type=float, default=1.4)
    parser.add_argument("--model", type=int, default=1)
    parser.add_argument("--residue-map", type=Path)
    parser.add_argument("--radii", type=Path)
    parser.add_argument("--ehma-carbons", default=",".join(f"C{i}" for i in range(7, 15)))
    parser.add_argument("--include-cavities", action="store_true")
    parser.add_argument("--max-grid-cells", type=int, default=40_000_000)
    parser.add_argument("--elev", type=float, default=25)
    parser.add_argument("--azim", type=float, default=-55)
    parser.add_argument("--detail-span", type=float, default=9, help="Local detail height, in Angstrom")
    parser.add_argument("--detail-center", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--pixels", type=int, default=1000, help="Pixels per whole-molecule panel")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--renderer", choices=("vtk", "matplotlib"), default="vtk")
    parser.add_argument("--interactive", action="store_true", help="Rotate synchronized views; Enter exports; Esc skips")
    parser.add_argument("--no-interactive", action="store_false", dest="interactive")
    parser.add_argument("--export-vtp", action="store_true", help="Also export the connected, cell-colored display mesh")
    parser.add_argument("--skip-patch-figure", action="store_true")
    parser.add_argument("--version", action="version", version=SCRIPT_VERSION)
    args = parser.parse_args()
    for name in ("grid_spacing", "detail_span"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not np.isfinite(args.probe_radius) or args.probe_radius < 0:
        parser.error("--probe-radius must be finite and non-negative")
    if not np.isfinite([args.elev, args.azim]).all():
        parser.error("Camera angles must be finite")
    if args.detail_center is not None and not np.isfinite(args.detail_center).all():
        parser.error("--detail-center values must be finite")
    if args.model < 1 or args.max_grid_cells < 8 or args.pixels < 100 or args.dpi < 50:
        parser.error("Invalid model, grid limit, pixel count or DPI")
    if args.interactive and args.renderer != "vtk":
        parser.error("Interactive preview requires --renderer vtk")
    if args.interactive and sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        parser.error("Interactive preview needs a desktop display; omit --interactive here")
    paths = []
    for pattern in args.pdb:
        matches = sorted(glob.glob(pattern))
        if not matches:
            parser.error(f"No PDB matches {pattern!r}")
        paths.extend(Path(p).resolve() for p in matches)
    paths = list(dict.fromkeys(paths))
    if len({p.stem for p in paths}) != len(paths):
        parser.error("Duplicate PDB filename stems; use separate output directories")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    for pdb in paths:
        surface = compute_surface(pdb, args)
        camera = initial_camera(surface, args)
        if args.interactive:
            camera = select_camera(surface, camera, args)
            if camera is None:
                print(f"Skipped {pdb.name}")
                continue
        detail = choose_detail(surface, camera, args)
        outputs = [compose_comparison(surface, camera, detail, args, "comparison")]
        patch_detail = None
        if not args.skip_patch_figure:
            patch_detail = choose_detail(surface, camera, args, label_kind="alkyl")
            outputs.append(compose_comparison(surface, camera, patch_detail, args, "patches"))
        save_analysis(surface, camera, detail, args, outputs, patch_detail)
        print(f"Saved {pdb.stem}: comparison, patch areas and original mesh", flush=True)
    print(f"Output: {args.out}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, RuntimeError, MemoryError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
