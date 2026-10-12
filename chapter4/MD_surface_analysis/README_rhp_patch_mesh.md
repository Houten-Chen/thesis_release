# RHP continuous surface and triangular mesh comparison

`render_rhp_patch_mesh.py` is a standalone script. Its analysis follows the algorithm in `rhp_surface_patches.py`, and its rendering uses the color scheme of the existing mosaic renderer.

## Running the script

```bash
python -m pip install numpy scipy scikit-image matplotlib pillow vtk
python render_rhp_patch_mesh.py seq151_1.pdb --out comparison
```

To process multiple PDB files in one run:

```bash
python render_rhp_patch_mesh.py seq151_1.pdb seq658_3.pdb --out comparison
```

To rotate the model on a desktop and confirm the viewing angle before saving:

```bash
python render_rhp_patch_mesh.py seq151_1.pdb --interactive --export-vtp
```

The two views share a camera. Drag the mouse to rotate, use the scroll wheel to zoom, press Enter to save, and press Esc to skip. Each PDB has a separate interactive preview.

The default grid spacing is 0.35 Å, and the probe radius is 1.4 Å. These can be changed with `--grid-spacing` and `--probe-radius`; the values should match those used for the analysis reported in the dissertation. `--elev 25 --azim -55` sets the initial viewing angle. `--detail-span 9` sets the height of the close-up region in Å. `--detail-center X Y Z` specifies its center.

## Calculations represented in the figures

1. All polymer atoms, including explicit hydrogens, define an unsmoothed external solvent-accessible surface (SAS). This is the boundary of the union of spheres whose radii are the atomic van der Waals radii plus the probe radius.
2. Each mesh vertex is assigned to the atom that minimizes the distance from the vertex to the atom center minus the expanded radius.
3. Each triangle contributes one-third of its area to each of its three vertices. For coloring, the original triangle is subdivided into six coplanar triangles. Each original vertex owns two of these, representing exactly one-third of the original triangle's area.
4. The EHMA alkyl selection includes C7–C14 and their attached hydrogens. Selected vertices are connected along the original mesh edges, and each connected component defines a patch. Its area is the sum of the area contributions assigned to its vertices.

Dark lines mark the original triangular mesh edges. In the close-up views, light lines show the equal-area regions assigned to the three vertices. Coplanar subdivision is used only for display; patch connectivity is always determined from the original mesh. The script applies no geometric smoothing or mesh decimation.

`*_comparison.png` compares continuous and triangular mesh views of the same surface side by side. Colors identify the entire monomer repeat unit. In `*_EHMA_patches.png`, the left panel highlights the EHMA alkyl surface, and the right panel uses different colors for connected patches. The remaining surface is light gray. This second figure directly illustrates the selection rules used to count patches.

## Outputs

For each PDB, the script produces the two comparison figures described above, each with close-up views, separate PNGs for the individual views, and the following files:

- `*_method.json` records the parameters, camera settings, mesh size, statistics, and area-conservation checks.
- `*_EHMA_patches.csv` records the area, vertex count, and associated residues of each patch.
- `*_surface_mesh.npz` stores the original analysis mesh and vertex labels.
- With `--export-vtp`, the script also writes `*_colored_surface.vtp`, a connected surface mesh with colors assigned to faces. This file can be opened in ParaView or similar software.

The script checks area conservation for each original triangle, the area allocation to each vertex, and the colored area of each patch. Example conformations illustrate the method; statistical distributions for each RHP should be calculated from the complete dataset.

On Linux without a desktop display, VTK 9.5 or later uses EGL rendering. Alternatively, add `--renderer matplotlib`, although this is slower for large meshes. On Windows and macOS, VTK desktop rendering can usually be used directly.
