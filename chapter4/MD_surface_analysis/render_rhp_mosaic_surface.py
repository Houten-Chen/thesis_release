#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render RHP PDBs as one continuous surface with discrete monomer colors.

Install:
    python -m pip install numpy scipy vtk matplotlib

Examples:
    python render_rhp_mosaic_surface.py seq151_1.pdb
    python render_rhp_mosaic_surface.py seq151_1.pdb seq658_1.pdb
    python render_rhp_mosaic_surface.py "seq*.pdb" --no-interactive

With no PDB arguments, use sequence 5, repeats 1--10, in each of the five
RHP groups; rank by mesh sphericity and open the best structure of each group.
Put this script beside the PDBs, or provide --pdb-dir.

Interactive controls:
    Left drag = rotate; wheel = zoom; Enter = save this view and continue.
    Escape / closing the window = skip this molecule.

Method (all distances are Angstroms):
    1. Build the union of atom spheres with radius vdW radius + probe radius.
    2. Optionally smooth the sphere-union implicit field (geometry only).
    3. Fill enclosed cavities and extract its exterior zero isosurface.
    4. Assign each triangle to the atom minimizing distance - inflated radius.
       Map that atom's whole monomer to a discrete RGB cell color.
       RGB colors are never interpolated between different monomer classes.

Default smoothing produces an illustrative, smoothed SAS-like envelope.
Use --smooth 0 for the unsmoothed exterior sphere-union surface. This is NOT
an exact solvent-excluded / Connolly surface. Smoothing changes surface areas
and patch boundaries; do not substitute the default illustrative mesh for
surface-patch statistics computed with a different surface definition.

Native VTK provides smooth lighting and an interactive rotation window.
--no-interactive defaults to the Matplotlib renderer (no OpenGL required).
The surface geometry and discrete triangle labels are identical in both.

Outputs after confirmation: *_mosaic.png, *_surface.vtp, *_view.json.
VTP retains mesh geometry, monomer labels, atom serials and RGB cell colors.
No PBC unwrapping is attempted. Provide a whole/unwrapped polymer snapshot.
Only the first PDB MODEL is read. Water and common free ions are excluded.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
    import vtk
    from scipy.ndimage import binary_fill_holes, gaussian_filter
    from scipy.spatial import cKDTree
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
except ImportError as exc:
    raise SystemExit(
        f"Missing dependency: {exc.name}\n"
        "Install: python -m pip install numpy scipy vtk matplotlib"
    ) from exc


# Edit these settings if the residue names / desired shades change.
SCRIPT_VERSION = "1.1"
MONOMERS = ("EHMA", "OEGMA", "SPMA", "MMA")
COLOR_HEX = {
    "EHMA": "#E53935",    # red
    "OEGMA": "#A6CEE3",   # light blue
    "SPMA": "#174A8B",    # dark blue
    "MMA": "#A6A6A6",     # gray
}
RESNAME_MAP = {
    "EH1": "EHMA", "EH2": "EHMA", "EHM": "EHMA",
    "HE1": "EHMA", "HE2": "EHMA", "HEH": "EHMA", "TEH": "EHMA",
    "OE1": "OEGMA", "OE2": "OEGMA", "OEG": "OEGMA",
    "HO1": "OEGMA", "HO2": "OEGMA", "TOE": "OEGMA",
    "SP1": "SPMA", "SP2": "SPMA", "SPM": "SPMA",
    "HS1": "SPMA", "HS2": "SPMA", "HSP": "SPMA", "TSP": "SPMA",
    "MM1": "MMA", "MM2": "MMA", "MMA": "MMA",
    "HM1": "MMA", "HM2": "MMA", "TMM": "MMA",
}
EXCLUDED_RESNAMES = {
    "HOH", "WAT", "SOL", "TIP", "T3P", "NA", "NA+", "K", "K+",
    "CL", "CL-", "CLA", "SOD", "POT", "CA", "MG", "ZN",
}
VDW_RADII_A = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52,
               "S": 1.80, "P": 1.80, "F": 1.47, "CL": 1.75,
               "BR": 1.85, "I": 1.98}
GROUPS = ("15", "30", "40", "50", "65")
PALETTE = np.array([
    [int(COLOR_HEX[m][i:i + 2], 16) for i in (1, 3, 5)] for m in MONOMERS
], dtype=np.uint8)


@dataclass
class Atoms:
    xyz: np.ndarray
    radii: np.ndarray
    monomer: np.ndarray
    serial: np.ndarray
    residues: dict[str, int]
    excluded_atoms: int


@dataclass
class Surface:
    pdb: Path
    atoms: Atoms
    mesh: object
    triangles: np.ndarray
    labels: np.ndarray
    metrics: dict


def read_pdb(path: Path, heavy_only: bool = False) -> Atoms:
    """Use fixed PDB columns; include all monomer atoms by default."""
    xyz, elements, labels, serials = [], [], [], []
    residue_keys, unknown = {}, Counter()
    excluded = 0
    started_model = False
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, 1):
            record = line[:6].strip()
            if record == "MODEL":
                if started_model or xyz:
                    break
                started_model = True
                continue
            if record == "ENDMDL":
                break
            if record not in {"ATOM", "HETATM"}:
                continue
            if line[16:17] not in {" ", "A", ""}:
                continue
            resname = line[17:20].strip().upper()
            if resname in EXCLUDED_RESNAMES:
                excluded += 1
                continue
            monomer = RESNAME_MAP.get(resname)
            if monomer is None:
                unknown[resname] += 1
                continue
            element = line[76:78].strip().upper()
            if not element:
                # These polymer atom names start with the element: C12, O3, H48.
                name = line[12:16].strip().lstrip("0123456789").upper()
                element = name[:2] if name.startswith(("CL", "BR")) else name[:1]
            if heavy_only and element == "H":
                continue
            if element not in VDW_RADII_A:
                raise ValueError(f"{path.name}:{line_no}: unknown element {element!r}")
            try:
                pos = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                serial = int(line[6:11])
            except ValueError as exc:
                raise ValueError(f"{path.name}:{line_no}: invalid PDB atom columns") from exc
            if not np.all(np.isfinite(pos)):
                raise ValueError(f"{path.name}:{line_no}: non-finite coordinates")
            xyz.append(pos)
            elements.append(element)
            labels.append(MONOMERS.index(monomer))
            serials.append(serial)
            residue_keys[(line[21:22], line[22:26], line[26:27], resname)] = resname
    if unknown:
        raise ValueError(
            f"{path.name}: unmapped residue names {dict(unknown)}. "
            "Add them to RESNAME_MAP near the top of this script."
        )
    if not xyz:
        raise ValueError(f"{path.name}: no polymer atoms found")
    return Atoms(np.asarray(xyz, dtype=float),
                 np.asarray([VDW_RADII_A[e] for e in elements]),
                 np.asarray(labels, dtype=np.uint8), np.asarray(serials, dtype=np.int32),
                 dict(Counter(residue_keys.values())), excluded)


def build_mesh(atoms: Atoms, spacing: float, probe: float, smooth: float,
               max_voxels: int) -> object:
    """Extract a single labeled-ready exterior envelope, not separate spheres."""
    radii = atoms.radii + probe
    band = max(3.0 * smooth + 2.0 * spacing, 2.0 * spacing)
    padding = float(radii.max()) + band + 2.0 * spacing
    origin = atoms.xyz.min(axis=0) - padding
    shape = np.ceil((atoms.xyz.max(axis=0) + padding - origin) / spacing).astype(int) + 1
    voxels = math.prod(int(s) for s in shape)
    if voxels > max_voxels:
        raise ValueError(
            f"Grid needs {voxels:,} voxels (limit {max_voxels:,}). "
            "Try --spacing 0.5, or check that the polymer is whole under PBC."
        )
    axes = [origin[i] + np.arange(shape[i]) * spacing for i in range(3)]
    field = np.full(tuple(shape), band, dtype=np.float32)
    for center, radius in zip(atoms.xyz, radii):
        reach = radius + band
        lo = np.maximum(np.floor((center - reach - origin) / spacing).astype(int), 0)
        hi = np.minimum(np.ceil((center + reach - origin) / spacing).astype(int) + 1, shape)
        dx = (axes[0][lo[0]:hi[0]] - center[0])[:, None, None]
        dy = (axes[1][lo[1]:hi[1]] - center[1])[None, :, None]
        dz = (axes[2][lo[2]:hi[2]] - center[2])[None, None, :]
        block = field[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        np.minimum(block, np.sqrt(dx * dx + dy * dy + dz * dz) - radius, out=block)
    if smooth > 0:
        gaussian_filter(field, sigma=smooth / spacing, output=field,
                        mode="nearest", truncate=3.0)
    inside = field < 0
    filled = binary_fill_holes(inside)
    field[filled & ~inside] = -spacing
    if not inside.any():
        raise ValueError("No negative implicit field; lower --smooth")
    image = vtk.vtkImageData()
    image.SetDimensions(*[int(s) for s in shape])
    image.SetOrigin(*origin)
    image.SetSpacing(spacing, spacing, spacing)
    values = numpy_to_vtk(field.ravel(order="F"), deep=True)
    values.SetName("sphere_union_field_A")
    image.GetPointData().SetScalars(values)
    contour = vtk.vtkFlyingEdges3D()
    contour.SetInputData(image)
    contour.SetValue(0, 0.0)
    contour.ComputeScalarsOff()
    contour.ComputeNormalsOff()
    contour.Update()
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(contour.GetOutputPort())
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.Update()
    mesh = vtk.vtkPolyData()
    mesh.DeepCopy(normals.GetOutput())
    if not mesh.GetNumberOfCells():
        raise ValueError("Surface extraction returned an empty mesh")
    return mesh


def mesh_triangles(mesh) -> np.ndarray:
    """Read VTK 9 cell storage without deprecated vtkCellArray.GetData()."""
    cells = mesh.GetPolys()
    offsets = vtk_to_numpy(cells.GetOffsetsArray())
    connectivity = vtk_to_numpy(cells.GetConnectivityArray())
    if (len(offsets) != cells.GetNumberOfCells() + 1
            or offsets[0] != 0 or offsets[-1] != len(connectivity)
            or not np.all(np.diff(offsets) == 3)
            or mesh.GetNumberOfCells() != cells.GetNumberOfCells()):
        raise ValueError("Expected a purely triangular surface")
    return connectivity.reshape(-1, 3)


def make_surface(path: Path, args) -> Surface:
    atoms = read_pdb(path, args.heavy_only)
    mesh = build_mesh(atoms, args.spacing, args.probe, args.smooth, args.max_voxels)
    verts = vtk_to_numpy(mesh.GetPoints().GetData())
    triangles = mesh_triangles(mesh)
    centers = verts[triangles].mean(axis=1)
    # Exact sphere owner: for each distinct radius, nearest center is its best
    # atom. Compare distance-radius across ALL radius classes (not just kNN).
    inflated = atoms.radii + args.probe
    best = np.full(len(centers), np.inf)
    owner = np.zeros(len(centers), dtype=np.int64)
    for radius in np.unique(inflated):
        indices = np.flatnonzero(inflated == radius)
        distance, local = cKDTree(atoms.xyz[indices]).query(centers)
        score = distance - radius
        update = score < best
        best[update] = score[update]
        owner[update] = indices[local[update]]
    labels = atoms.monomer[owner]
    for name, data in (("monomer_id", labels), ("owner_atom_serial", atoms.serial[owner])):
        array = numpy_to_vtk(data, deep=True)
        array.SetName(name)
        mesh.GetCellData().AddArray(array)
    colors = numpy_to_vtk(PALETTE[labels], deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    colors.SetName("monomer_rgb")
    mesh.GetCellData().SetScalars(colors)
    names = vtk.vtkStringArray()
    names.SetName("monomer_id_names")
    for name in MONOMERS:
        names.InsertNextValue(name)
    mesh.GetFieldData().AddArray(names)
    mass = vtk.vtkMassProperties()
    mass.SetInputData(mesh)
    mass.Update()
    area, volume = mass.GetSurfaceArea(), mass.GetVolume()
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(mesh)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.Update()
    regions = connectivity.GetNumberOfExtractedRegions()
    metrics = {"atoms": len(atoms.xyz), "vertices": len(verts),
               "triangles": len(triangles), "surface_components": regions,
               "surface_area_A2": float(area), "enclosed_volume_A3": float(volume),
               "sphericity": float((36.0 * math.pi * volume * volume) ** (1.0 / 3.0) / area)}
    print(f"{path.name}: {len(atoms.xyz)} atoms, {len(triangles):,} triangles, "
          f"{regions} component(s), sphericity={metrics['sphericity']:.4f}", flush=True)
    if regions > 1:
        print("  Multiple surface components are retained; check the whole/PBC coordinates.")
    return Surface(path, atoms, mesh, triangles, labels, metrics)


def save_assets(surface: Surface, args, camera: dict) -> None:
    outbase = args.out / surface.pdb.stem
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(outbase) + "_surface.vtp")
    writer.SetInputData(surface.mesh)
    writer.SetDataModeToAppended()
    writer.SetCompressorTypeToZLib()
    if writer.Write() != 1:
        raise OSError("Could not save the VTP mesh")
    metadata = {
        "pdb": surface.pdb.name, "coordinate_units": "Angstrom",
        "surface_definition": "exterior of smoothed union of vdW+probe atom spheres",
        "grid_spacing_A": args.spacing, "probe_radius_A": args.probe,
        "field_smoothing_sigma_A": args.smooth, "heavy_atoms_only": args.heavy_only,
        "label_assignment": "triangle centroid: argmin(distance_to_atom - vdW - probe)",
        "monomer_id_names": list(MONOMERS), "colors": COLOR_HEX,
        "residue_counts": surface.atoms.residues, "excluded_atoms": surface.atoms.excluded_atoms,
        "metrics": surface.metrics, "camera": camera,
    }
    Path(str(outbase) + "_view.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved: {surface.pdb.stem}_mosaic.png (+ surface.vtp and view.json)", flush=True)


def create_vtk_scene(surface: Surface, args):
    """Build the full native scene without starting OpenGL or an event loop."""
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)
    renderer.GradientBackgroundOff()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(surface.mesh)
    mapper.SetScalarModeToUseCellData()
    mapper.SetColorModeToDirectScalars()
    mapper.ScalarVisibilityOn()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetInterpolationToPhong()
    prop.SetAmbient(0.38)
    prop.SetDiffuse(0.62)
    prop.SetSpecular(0.10)
    prop.SetSpecularPower(28)
    prop.EdgeVisibilityOff()
    renderer.AddViewProp(actor)
    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    window.SetWindowName(f"RHP mosaic - {surface.pdb.name} - Enter to save")
    window.SetSize(900, 900)
    window.SetMultiSamples(8)
    if args.no_interactive:
        window.SetOffScreenRendering(1)
    renderer.ResetCamera()
    camera = renderer.GetActiveCamera()
    camera.ParallelProjectionOn()
    camera.Azimuth(args.azim)
    camera.Elevation(args.elev)
    camera.OrthogonalizeViewUp()
    renderer.ResetCameraClippingRange()
    title = vtk.vtkTextActor()
    title.SetInput(surface.pdb.name)
    title.SetPosition(20, 855)
    title.GetTextProperty().SetColor(0.15, 0.15, 0.15)
    title.GetTextProperty().SetFontSize(20)
    controls = vtk.vtkTextActor()
    controls.SetInput("Drag: rotate   Wheel: zoom   Enter: save   Esc: skip")
    controls.SetPosition(20, 20)
    controls.GetTextProperty().SetColor(0.25, 0.25, 0.25)
    controls.GetTextProperty().SetFontSize(15)
    # AddViewProp accepts both 2D and 3D props and remains available in
    # versions where the deprecated AddActor2D method has been removed.
    renderer.AddViewProp(title)
    renderer.AddViewProp(controls)
    legend = vtk.vtkLegendBoxActor()
    legend.SetNumberOfEntries(len(MONOMERS))
    symbol = vtk.vtkPlaneSource()
    symbol.Update()
    for i, name in enumerate(MONOMERS):
        legend.SetEntry(i, symbol.GetOutput(), name, PALETTE[i].astype(float) / 255.0)
    legend.SetPosition(0.02, 0.76)
    legend.SetPosition2(0.20, 0.17)
    legend.BorderOff()
    legend.BoxOff()
    legend.GetEntryTextProperty().SetColor(0.15, 0.15, 0.15)
    renderer.AddViewProp(legend)
    return window, camera, title, controls, legend


def render_vtk(surface: Surface, args) -> bool:
    window, camera, title, controls, legend = create_vtk_scene(surface, args)
    saved = False

    def save() -> None:
        nonlocal saved
        title.VisibilityOff()
        controls.VisibilityOff()
        legend.SetVisibility(args.legend)
        window.Render()
        capture = vtk.vtkWindowToImageFilter()
        capture.SetInput(window)
        capture.SetInputBufferTypeToRGB()  # opaque white, no alpha background
        capture.ReadFrontBufferOff()
        capture.SetScale(args.scale)
        capture.Modified()
        capture.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(args.out / f"{surface.pdb.stem}_mosaic.png"))
        writer.SetInputConnection(capture.GetOutputPort())
        writer.Write()
        save_assets(surface, args, {
            "renderer": "VTK", "position": list(camera.GetPosition()),
            "focal_point": list(camera.GetFocalPoint()), "view_up": list(camera.GetViewUp()),
            "parallel_scale": camera.GetParallelScale(), "parallel_projection": True,
        })
        saved = True

    try:
        if args.no_interactive:
            window.Render()
            save()
        else:
            interactor = vtk.vtkRenderWindowInteractor()
            interactor.SetRenderWindow(window)
            interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

            def key_press(obj, _event):
                if obj.GetKeySym() in {"Return", "KP_Enter"}:
                    save()
                    obj.TerminateApp()
                elif obj.GetKeySym() == "Escape":
                    obj.TerminateApp()

            interactor.AddObserver("KeyPressEvent", key_press, 1.0)
            interactor.Initialize()
            window.Render()
            interactor.Start()
    finally:
        window.Finalize()
    return saved


def render_matplotlib(surface: Surface, args) -> bool:
    import matplotlib
    if args.no_interactive:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    verts = vtk_to_numpy(surface.mesh.GetPoints().GetData())
    normals = vtk_to_numpy(surface.mesh.GetPointData().GetNormals())
    # Smooth geometry lighting, but one discrete monomer color per triangle.
    face_normals = normals[surface.triangles].mean(axis=1)
    face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-12)
    light = np.array([-0.45, -0.55, 0.80])
    light /= np.linalg.norm(light)
    shade = 0.58 + 0.42 * np.maximum(face_normals @ light, 0.0)
    colors = PALETTE[surface.labels].astype(float) / 255.0 * shade[:, None]
    fig = plt.figure(figsize=(6, 6), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], projection="3d", facecolor="white")
    collection = Poly3DCollection(verts[surface.triangles], facecolors=colors,
                                 edgecolors="none", linewidths=0, antialiased=False)
    ax.add_collection3d(collection)
    center = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
    half = np.max(np.ptp(verts, axis=0)) * 0.53
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1), zoom=1.17)
    ax.set_proj_type("ortho")
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_axis_off()
    title = fig.text(0.03, 0.95, surface.pdb.name, color="#333333")
    controls = fig.text(0.5, 0.025, "Drag: rotate   Enter: save   Esc: skip",
                        ha="center", color="#444444", fontsize=9)
    legend = fig.legend(handles=[Patch(facecolor=COLOR_HEX[m], label=m) for m in MONOMERS],
                        loc="upper left", bbox_to_anchor=(0.02, 0.92), frameon=False)
    saved = False

    def save():
        nonlocal saved
        title.set_visible(False)
        controls.set_visible(False)
        legend.set_visible(args.legend)
        fig.savefig(args.out / f"{surface.pdb.stem}_mosaic.png",
                    dpi=150 * args.scale, facecolor="white", transparent=False)
        save_assets(surface, args, {"renderer": "Matplotlib", "elev": float(ax.elev),
                                   "azim": float(ax.azim), "roll": float(ax.roll),
                                   "xlim": list(ax.get_xlim()), "ylim": list(ax.get_ylim()),
                                   "zlim": list(ax.get_zlim()), "projection": "ortho"})
        saved = True

    if args.no_interactive:
        save()
        plt.close(fig)
    else:
        def key_press(event):
            if event.key == "enter":
                save()
                plt.close(fig)
            elif event.key == "escape":
                plt.close(fig)
        fig.canvas.mpl_connect("key_press_event", key_press)
        plt.show(block=True)
    return saved


def select_seq5(args):
    """Require all 10 repeats per group, then rank their surface sphericities."""
    candidates = {
        group: [args.pdb_dir / f"seq{group}5_{i}.pdb" for i in range(1, 11)]
        for group in GROUPS
    }
    missing = [p.name for paths in candidates.values() for p in paths if not p.is_file()]
    if missing:
        raise ValueError(
            "No-argument mode needs sequence 5, repeats 1--10, for all five groups. "
            f"Missing {len(missing)} files; examples: {', '.join(missing[:5])}. "
            "For a single example, give its filename as an argument."
        )
    import csv
    # Save the candidate scores even when some interactive views are skipped.
    with (args.out / "seq5_sphericity_selection.csv").open("w", newline="", encoding="utf-8") as h:
        fields = ["group", "pdb", "sphericity", "surface_components", "selected"]
        table = csv.DictWriter(h, fieldnames=fields)
        table.writeheader()
        for group, paths in candidates.items():
            best = None
            group_rows = []
            for path in paths:
                current = make_surface(path, args)
                group_rows.append({"group": group, "pdb": path.name,
                                   "sphericity": current.metrics["sphericity"],
                                   "surface_components": current.metrics["surface_components"]})
                if best is None or current.metrics["sphericity"] > best.metrics["sphericity"]:
                    best = current
            for row in group_rows:
                row["selected"] = int(row["pdb"] == best.pdb.name)
                table.writerow(row)
            h.flush()
            print(f"Selected RHP group {group}: {best.pdb.name}", flush=True)
            yield best


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {SCRIPT_VERSION} (VTK {vtk.vtkVersion.GetVTKVersion()})")
    parser.add_argument("pdb", nargs="*", help="PDB filenames; quoted wildcards are supported")
    parser.add_argument("--pdb-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="PDB folder for no-argument sequence-5 selection")
    parser.add_argument("--out", type=Path, default=Path("rhp_mosaic_output"))
    parser.add_argument("--spacing", type=float, default=0.35, help="Grid spacing in Angstroms")
    parser.add_argument("--probe", type=float, default=1.4, help="Probe radius in Angstroms")
    parser.add_argument("--smooth", type=float, default=0.65, help="Geometry smoothing sigma in Angstroms; 0 disables")
    parser.add_argument("--heavy-only", action="store_true", help="Omit hydrogens; default includes all polymer atoms")
    parser.add_argument("--no-interactive", action="store_true", help="Save initial view automatically")
    parser.add_argument("--renderer", choices=("auto", "vtk", "matplotlib"), default="auto")
    parser.add_argument("--legend", action="store_true", help="Keep the monomer legend in exported PNGs")
    parser.add_argument("--scale", type=int, default=3, help="PNG export scale; 3 produces 2700 x 2700 pixels")
    parser.add_argument("--elev", type=float, default=25.0, help="Initial camera elevation in degrees")
    parser.add_argument("--azim", type=float, default=-55.0, help="Initial camera azimuth in degrees")
    parser.add_argument("--max-voxels", type=int, default=64_000_000, help="Grid memory guard")
    args = parser.parse_args()
    if not np.isfinite(args.spacing) or args.spacing <= 0:
        parser.error("--spacing must be finite and positive")
    if not np.isfinite(args.probe) or not np.isfinite(args.smooth) or args.probe < 0 or args.smooth < 0:
        parser.error("--probe and --smooth must be finite and nonnegative")
    if args.scale < 1 or args.max_voxels < 1:
        parser.error("--scale and --max-voxels must be positive")
    if not np.isfinite(args.elev) or not np.isfinite(args.azim):
        parser.error("Camera angles must be finite")
    args.out = args.out.resolve()
    args.pdb_dir = args.pdb_dir.resolve()
    if args.renderer == "auto":
        args.renderer = "matplotlib" if args.no_interactive else "vtk"
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        if not args.no_interactive:
            parser.error("No graphical display. Use --no-interactive, or run on your desktop.")
        if args.renderer == "vtk" and not hasattr(vtk, "vtkEGLRenderWindow"):
            parser.error("This VTK build requires a display; use --renderer matplotlib")
    return args


def main():
    args = parse_args()
    print(f"RHP mosaic renderer {SCRIPT_VERSION} | VTK {vtk.vtkVersion.GetVTKVersion()}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.pdb:
        paths = []
        for pattern in args.pdb:
            expanded = sorted(glob.glob(pattern))
            if not expanded:
                raise ValueError(f"No PDB matches {pattern!r}")
            paths.extend(Path(p).resolve() for p in expanded)
        paths = list(dict.fromkeys(paths))
        stems = [p.stem for p in paths]
        if len(set(stems)) != len(stems):
            raise ValueError("Inputs have duplicate filename stems; render them into separate output folders")
        surfaces = (make_surface(p, args) for p in paths)
    else:
        surfaces = select_seq5(args)
    saved = skipped = 0
    render = render_vtk if args.renderer == "vtk" else render_matplotlib
    for surface in surfaces:
        if render(surface, args):
            saved += 1
        else:
            skipped += 1
            print(f"Skipped: {surface.pdb.name}")
    print(f"Finished: {saved} saved, {skipped} skipped. Output: {args.out}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
