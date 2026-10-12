#!/usr/bin/env python3
"""Analyze ONE RHP PDB: monomer surface composition and EHMA alkyl patches.

Install: python -m pip install numpy scipy scikit-image
Example: python rhp_surface_patches.py seq658_1.pdb --out seq658_1_surface

Geometry is the external solvent-accessible surface (SAS), the boundary of
the union of spheres with radius (atomic VdW radius + probe radius). It is NOT
the solvent-excluded surface (SES). Explicit hydrogens are retained.

EHMA alkyl means C7..C14 in EH1/EH2/TEH (and configured EHMA residues), plus
hydrogens attached to those carbons. Ester C6/O1/O2 and backbone are excluded.
Hydrophobicity here is a chemical/geometric label, not a hydration free energy.

Independent implementation informed by the publicly inspected source:
  MolPatch: ibivu/hydrophobic_patches, commit
    be919dca3bd35c6d3be1000e30de4e4bf98af625
    src/molpatch/AtomBased/{ProteinPatch,Patch}.py
  PEP-Patch: liedllab/surface_analyses, commit
    162a7c822acdedc9f281a7cc68efe88d8e33928f
    surface_analyses/{patches,surface,hydrophobic_potential}.py

Uses PEP-Patch's definition of mesh-edge connectivity and barycentric area
allocation (one third of each triangle to each vertex). Unlike whole-atom
area aggregation, each surface area contribution is counted only once.

This initial script has NOT been run on the example PDBs. Grid convergence
has NOT been measured. Try 0.50, 0.35, 0.25 Angstrom on one PDB first.
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

VERSION = "0.1.0-unrun"
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


def patch_statistics(rows, min_area):
    areas = np.array([r["area_A2"] for r in rows], dtype=float)
    kept = areas[areas >= min_area]
    return {
        "EHMA_patch_count_all": len(areas),
        "EHMA_patch_mean_all_A2": float(areas.mean()) if len(areas) else None,
        "EHMA_patch_area_weighted_mean_A2": (float(areas @ areas / areas.sum())
                                            if len(areas) and areas.sum() > 0 else None),
        "EHMA_largest_patch_A2": float(areas.max()) if len(areas) else 0.0,
        "min_patch_area_filter_A2": min_area,
        "EHMA_patch_count_filtered": len(kept),
        "EHMA_patch_mean_filtered_A2": float(kept.mean()) if len(kept) else None,
        "EHMA_patch_area_filtered_A2": float(kept.sum()),
        "EHMA_patch_area_below_filter_A2": float(areas[areas < min_area].sum()),
    }


def write_csv(path, rows, columns=None):
    if columns is None:
        columns = list(rows[0]) if rows else ["patch_id", "area_A2", "area_nm2"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_ply(path, vertices, faces, owner, patch_id, atoms, by_patch=False):
    """ASCII PLY, coordinates in Angstrom; scalar EHMA patch ID is preserved."""
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\ncomment coordinates in Angstrom\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("property int monomer_id\nproperty int EHMA_patch_id\n")
        handle.write(f"element face {len(faces)}\n")
        handle.write("property list uchar int vertex_indices\nend_header\n")
        palette = [(230, 96, 43), (52, 120, 191), (140, 72, 173),
                   (207, 151, 39), (45, 154, 141), (202, 69, 117)]
        for xyz, atom_index, pid in zip(vertices, owner, patch_id):
            monomer = atoms[atom_index].monomer
            color = ((palette[(int(pid) - 1) % len(palette)] if pid > 0 else (210, 210, 210))
                     if by_patch else COLORS[monomer])
            handle.write(f"{xyz[0]:.5f} {xyz[1]:.5f} {xyz[2]:.5f} "
                         f"{color[0]} {color[1]} {color[2]} "
                         f"{MONOMERS.index(monomer)} {int(pid)}\n")
        for a, b, c in faces:
            handle.write(f"3 {a} {b} {c}\n")


def analyze(args):
    started = time.perf_counter()
    residue_map = RESIDUE_MAP | {str(k).upper(): str(v).upper()
                                 for k, v in read_json_object(args.residue_map).items()}
    if any(v not in MONOMERS for v in residue_map.values()):
        raise ValueError("--residue-map values must be MMA, EHMA, OEGMA or SPMA.")
    radii = VDW_RADII | {str(k).upper(): float(v)
                         for k, v in read_json_object(args.radii).items()}
    if any(not np.isfinite(v) or v <= 0 for v in radii.values()):
        raise ValueError("All VdW radii must be finite and positive (Angstrom).")
    carbon_names = {name.strip() for name in args.ehma_carbons.split(",") if name.strip()}
    if not carbon_names:
        raise ValueError("--ehma-carbons must list at least one carbon name.")
    atoms, warnings = read_pdb(args.pdb, residue_map, args.model)
    xyz = np.array([a.xyz for a in atoms], dtype=np.float64)
    if not np.isfinite(xyz).all():
        raise ValueError("PDB contains non-finite coordinates.")
    residues, max_bond = classify_and_check(atoms, carbon_names)
    atomic_radii = np.array([radii[a.element] for a in atoms])
    print(f"Input: {len(atoms)} atoms, {len(residues)} repeat units. "
          f"Grid spacing: {args.grid_spacing:g} A.", flush=True)
    vertices, faces, vertex_area, grid_info = make_sas(
        xyz, atomic_radii + args.probe_radius, args.grid_spacing,
        args.max_grid_cells, args.include_cavities)
    owner, surface_error = assign_owner(vertices, xyz, atomic_radii + args.probe_radius)
    atom_area = np.bincount(owner, weights=vertex_area, minlength=len(atoms))
    patch_id, patches = find_ehma_patches(vertices, faces, vertex_area, owner, atoms)
    total_area = float(vertex_area.sum())
    ehma_area = float(sum(atom_area[i] for i, a in enumerate(atoms) if a.ehma_alkyl))
    if not np.isclose(sum(r["area_A2"] for r in patches), ehma_area, rtol=1e-10, atol=1e-7):
        raise RuntimeError("EHMA patch areas do not conserve EHMA alkyl surface area.")
    monomer_rows = []
    for monomer in MONOMERS:
        ids = [i for i, a in enumerate(atoms) if a.monomer == monomer]
        n_units = sum(atoms[rs[0]].monomer == monomer for rs in residues.values())
        area = float(atom_area[ids].sum())
        row = {"monomer": monomer, "n_repeat_units": n_units,
               "repeat_unit_fraction": n_units / len(residues),
               "surface_area_A2": area, "surface_fraction": area / total_area,
               "surface_percent": 100 * area / total_area}
        for region in ("backbone_common", "ester", "sidechain"):
            row[region + "_surface_A2"] = float(sum(atom_area[i] for i in ids
                                                    if atoms[i].region == region))
        monomer_rows.append(row)
    residue_rows = [{"residue": atoms[ids[0]].residue_label,
                     "resname": atoms[ids[0]].resname, "monomer": atoms[ids[0]].monomer,
                     "surface_area_A2": float(atom_area[ids].sum()),
                     "EHMA_alkyl_surface_A2": float(sum(atom_area[i] for i in ids
                                                        if atoms[i].ehma_alkyl))}
                    for ids in residues.values()]
    atom_rows = [{"atom_serial": a.serial, "atom_name": a.name, "element": a.element,
                  "residue": a.residue_label, "monomer": a.monomer, "region": a.region,
                  "parent_heavy_atom_serial": atoms[a.parent].serial,
                  "is_EHMA_alkyl": int(a.ehma_alkyl), "vdw_radius_A": float(atomic_radii[i]),
                  "surface_area_A2": float(atom_area[i])} for i, a in enumerate(atoms)]
    for row in patches:
        row["passes_area_filter"] = int(row["area_A2"] >= args.min_patch_area)
    summary = {
        "script_version": VERSION, "input_pdb": str(args.pdb.resolve()),
        "model_occurrence": args.model,
        "surface": "SAS (not SES)", "external_surface_only": not args.include_cavities,
        "units": "coordinates Angstrom; area Angstrom^2; 100 Angstrom^2 = 1 nm^2",
        "probe_radius_A": args.probe_radius, "grid_spacing_A": args.grid_spacing,
        "vdw_radii_A": radii, "residue_map": residue_map,
        "EHMA_alkyl_carbons": sorted(carbon_names),
        "EHMA_hydrogens": "included if attached to selected alkyl carbons",
        "area_allocation": "one third of each triangle to each vertex, radius-aware atom ownership",
        "patch_connectivity": "selected vertices connected along mesh edges; no distance cutoff",
        "PBC_policy": "require whole coordinates; no image correction; analyze isolated polymer",
        "hydrophobicity_policy": "EHMA alkyl chemical label; not total apolar surface or hydration energy",
        "n_atoms": len(atoms), "n_hydrogens": sum(a.element == "H" for a in atoms),
        "n_repeat_units": len(residues), "max_backbone_bond_A": max_bond,
        "total_surface_area_A2": total_area, "EHMA_alkyl_surface_A2": ehma_area,
        "EHMA_alkyl_surface_fraction": ehma_area / total_area,
        "max_vertex_distance_from_exact_SAS_A": surface_error,
        **patch_statistics(patches, args.min_patch_area), **grid_info,
        "resolution_convergence_verified": False,
        "warnings": warnings, "elapsed_seconds": time.perf_counter() - started,
    }
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                               allow_nan=False) + "\n", encoding="utf-8")
    flat = {k: v for k, v in summary.items()
            if isinstance(v, (str, int, float, bool)) or v is None}
    write_csv(out / "summary.csv", [flat])
    write_csv(out / "monomer_surface.csv", monomer_rows)
    write_csv(out / "residue_surface.csv", residue_rows)
    write_csv(out / "atom_surface.csv", atom_rows)
    patch_columns = ["patch_id", "area_A2", "area_nm2", "n_vertices", "n_contributing_atoms",
                     "n_EHMA_units", "EHMA_residues", "center_x_A", "center_y_A", "center_z_A",
                     "passes_area_filter"]
    write_csv(out / "EHMA_patches.csv", patches, patch_columns)
    np.savez_compressed(out / "surface_mesh.npz", vertices_A=vertices, faces=faces,
                        vertex_area_A2=vertex_area, owner_atom_index=owner,
                        EHMA_patch_id=patch_id, atom_serial=np.array([a.serial for a in atoms]),
                        atom_monomer=np.array([a.monomer for a in atoms]),
                        atom_is_EHMA_alkyl=np.array([a.ehma_alkyl for a in atoms]))
    if args.export_ply:
        write_ply(out / "surface_monomers.ply", vertices, faces, owner, patch_id, atoms)
        write_ply(out / "surface_EHMA_patches.ply", vertices, faces, owner, patch_id,
                  atoms, by_patch=True)
    print(f"Total SAS: {total_area:.2f} A^2; EHMA alkyl: {ehma_area:.2f} A^2; "
          f"patches: {len(patches)}. Outputs: {out.resolve()}", flush=True)
    for warning in warnings:
        print("WARNING: " + warning, file=sys.stderr)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdb", type=Path, help="ONE intact RHP PDB file")
    parser.add_argument("--out", type=Path, default=None, help="Output folder; default <PDB stem>_surface")
    parser.add_argument("--grid-spacing", type=float, default=0.5, help="Grid spacing in A (initial default 0.5; test convergence)")
    parser.add_argument("--probe-radius", type=float, default=1.4, help="Solvent probe radius in A (default 1.4)")
    parser.add_argument("--min-patch-area", type=float, default=10.0, help="Secondary size filter in A^2; raw patch statistics are also kept (default 10)")
    parser.add_argument("--model", type=int, default=1, help="1-based MODEL occurrence (default 1)")
    parser.add_argument("--residue-map", type=Path, help='JSON residue mapping, e.g. {"EH1": "EHMA"}')
    parser.add_argument("--radii", type=Path, help='JSON VdW radii overrides in A, e.g. {"C": 1.7}')
    parser.add_argument("--ehma-carbons", default=",".join(f"C{i}" for i in range(7, 15)), help="Comma-separated EHMA alkyl carbon names")
    parser.add_argument("--include-cavities", action="store_true", help="Also count sealed internal probe-sized cavity surfaces")
    parser.add_argument("--max-grid-cells", type=int, default=40_000_000, help="Grid allocation limit (default 40000000 cells)")
    parser.add_argument("--export-ply", action="store_true", help="Also write monomer-colored and EHMA patch-colored PLY meshes")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing output files in an existing analysis folder")
    args = parser.parse_args()
    if not args.pdb.is_file():
        parser.error("Input PDB does not exist.")
    if args.out is None:
        args.out = args.pdb.with_name(args.pdb.stem + "_surface")
    if args.out.exists() and not args.out.is_dir():
        parser.error("--out exists and is not a directory.")
    if not (np.isfinite(args.grid_spacing) and args.grid_spacing > 0
            and np.isfinite(args.probe_radius) and args.probe_radius >= 0
            and np.isfinite(args.min_patch_area) and args.min_patch_area >= 0
            and args.model >= 1 and args.max_grid_cells >= 8):
        parser.error("Invalid grid/probe/filter/model/grid-limit argument.")
    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        parser.error("Output folder is not empty; choose a new --out or use --overwrite.")
    try:
        analyze(args)
    except (ValueError, KeyError, MemoryError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
