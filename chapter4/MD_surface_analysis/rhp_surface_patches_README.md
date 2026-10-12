# Surface composition and EHMA patch analysis for a single RHP PDB

`rhp_surface_patches.py` analyzes one PDB file. Its design draws on the core source code of two public repositories and the naming conventions in the supplied PDB files. At the initial delivery, PDB analysis and mesh-convergence validation had not yet been performed.

## Installation and usage

Python 3.10 or later is recommended.

```bash
python -m pip install numpy scipy scikit-image
python rhp_surface_patches.py seq658_1.pdb --out seq658_1_surface --grid-spacing 0.5 --export-ply
```

Enclose paths containing spaces in double quotes. `--export-ply` is optional. The default output directory is `<filename>_surface` in the same directory as the input PDB. Existing outputs are preserved by default; add `--overwrite` to replace them.

## What the script calculates

1. All polymer atoms, including explicit hydrogens present in the file, are used to construct the surface. Water and common ions are ignored.
2. The surface is the **solvent-accessible surface (SAS)**, defined as the boundary of the union of spheres whose radii are the atomic van der Waals radii plus the probe radius. The default probe radius is 1.4 Å.
3. A triangular mesh represents this surface. By default, enclosed internal cavities inaccessible to probe centers from the exterior are excluded, while open grooves and channels are retained. Connectivity to the exterior is determined on the voxel grid and therefore also depends on grid spacing.
4. Each vertex is assigned to the atom that minimizes `distance from the vertex to the atom center - expanded radius`. This assignment accounts for differences in atomic radii, rather than relying solely on the nearest atom center.
5. Each triangle contributes one-third of its area to each vertex. These contributions are then summed by atom and monomer. The areas of the four monomer classes therefore sum to the total surface area.
6. Vertices assigned to EHMA alkyl carbons and their hydrogens are labeled as EHMA alkyl. They are connected along mesh edges, and each connected region defines a patch. No additional spatial-distance cutoff is used.
7. Each patch area is the sum of the areas assigned to its vertices. The sum of all EHMA patch areas equals the exposed EHMA alkyl area. Triangles with mixed vertex labels are partitioned according to the fraction of vertices carrying each label. This boundary approximation changes as the mesh is refined.

Here, a patch is a **continuous geometrically exposed region contributed by EHMA alkyl groups**. The definition excludes mixed hydrophobic patches connected through other monomers, such as MMA. Hydrophobicity is not defined through hydration free energies or a logP-based potential field.

**SAS and the solvent-excluded surface (SES) are different surface definitions.** Areas calculated by this script should be compared with results using the same definition. This analysis is not a numerical reproduction of the original MolPatch or PEP-Patch workflows.

## Supported PDB naming conventions

| Monomer | Residue names |
| --- | --- |
| MMA | MM1, MM2, HM1, HM2, TMM |
| EHMA | EH1, EH2, HE1, HE2, TEH |
| OEGMA | OE1, OE2, HO1, HO2, TOE |
| SPMA | SP1, SP2, HS1, HS2, TSP |

Heavy-atom names were checked for the residues present in the supplied files: MM1/MM2/HM2/TMM, EH1/EH2/TEH, OE1/OE2/HO1/TOE, and SP1/SP2. Other terminal aliases are configurable candidates; their chemical correspondence should be confirmed before use. Terminal residues are included in the repeat-unit count of their assigned monomer.

The EHMA alkyl selection is `C7,C8,C9,C10,C11,C12,C13,C14`. Each hydrogen is assigned to the nearest heavy atom within the same residue. Only hydrogens assigned to these carbons contribute to EHMA patches. `C6/O1/O2` belong to the ester group and are excluded from the alkyl selection.

`backbone_common` includes C1–C5 and their attached hydrogens. It covers the polymer backbone, the common methyl group, and terminal carbons, so it is not strictly a backbone-only area. `ester` contains C6/O1/O2, and `sidechain` contains the remaining groups. Monomer surface fractions use the area of the entire repeat unit; EHMA patches use the alkyl side-chain area.

If residue names differ, create a JSON mapping file, for example:

```json
{"MYE": "EHMA", "MYO": "OEGMA"}
```

Then add `--residue-map mapping.json`. If EHMA carbon names differ, use `--ehma-carbons C7,C8,C9,C10,C11,C12,C13,C14` with the actual names. Unknown residues cause an error to prevent silent misclassification.

## Output files

| File | Contents |
| --- | --- |
| `summary.json` | Parameters, surface definition, atom counts, mesh information, statistics for all patches, and warnings |
| `summary.csv` | A single-row numerical summary for inspection |
| `monomer_surface.csv` | Actual monomer number fractions, surface-area fractions, and group areas for the four monomer classes |
| `EHMA_patches.csv` | Area, associated EHMA residues, and area-weighted centroid of each patch, sorted by decreasing area |
| `residue_surface.csv` | Exposed area and exposed EHMA alkyl area for each repeat unit |
| `atom_surface.csv` | Area and labels for each atom, plus the heavy-atom assignment of each hydrogen, for checking the selection rules |
| `surface_mesh.npz` | Triangular mesh, vertex areas, source atoms, and EHMA patch IDs; coordinates are in Å |
| `surface_monomers.ply` | Optional mesh colored by the four monomer classes |
| `surface_EHMA_patches.ply` | Optional mesh with each EHMA patch colored separately and the remaining surface shown in gray |

PLY files can be opened in software that supports vertex colors, such as MeshLab. CSV files use UTF-8 with a byte-order mark (BOM) for convenient opening in Excel. The script processes one selected MODEL, defaulting to the first MODEL in the PDB. Use `--model 2` to select the second.

## Mean patch area

Let the area of patch j be `A_j`.

- `EHMA_patch_mean_all_A2` is the arithmetic mean, `sum(A_j) / patch_count`.
- `EHMA_patch_area_weighted_mean_A2` is the area-weighted mean, `sum(A_j**2) / sum(A_j)`. When a point is sampled uniformly from the exposed EHMA surface, this is the expected area of the patch containing that point.
- `EHMA_largest_patch_A2` is the area of the largest patch.
- The script also records the count and arithmetic mean of patches meeting `--min-patch-area`, which defaults to 10 Å². This is an **adjustable analysis threshold** for examining the influence of small fragments; it has not been validated as a physical threshold for RHPs. All original patches are retained.

If no patches are exposed, the arithmetic and area-weighted means are null in JSON and empty in CSV. The largest patch area and patch count are zero.

All area statistics are in Å²; 100 Å² = 1 nm². Surface-area percentages and monomer number percentages describe different quantities, particularly because OEGMA has a larger side chain. Their ratio should not be interpreted directly as surface enrichment corrected for size effects.

## Checking mesh resolution locally

This method uses a surface mesh, so grid-spacing convergence should be checked. First, run the analysis on **the same PDB with the same atomic radii and probe radius** at several grid spacings:

```bash
python rhp_surface_patches.py seq658_1.pdb --out scan_050 --grid-spacing 0.50
python rhp_surface_patches.py seq658_1.pdb --out scan_035 --grid-spacing 0.35
python rhp_surface_patches.py seq658_1.pdb --out scan_025 --grid-spacing 0.25
```

Compare the total area, surface-area percentage of each monomer class, EHMA alkyl area, largest patch area, area-weighted mean, and patch counts and arithmetic means both before and after area filtering.

If the total area is stable but patch counts or arithmetic means still change substantially, inspect the resolution of chemical boundaries and narrow connecting channels. Finer meshes require more memory and computation time, and a sufficient grid spacing cannot be established in advance. The initial default of 0.5 Å is a starting point for local trial runs, not a validated optimal resolution.

`--probe-radius` changes the geometric accessibility scale and therefore the physical surface definition. It is distinct from `--grid-spacing`, which is varied to assess numerical convergence.

## Coordinate and parameter limitations

- Input coordinates should describe a complete, continuous polymer. The script does not reconstruct periodic-boundary images automatically. It checks distance-based covalent connectivity within each residue and C3–C2 backbone connections between adjacent residues, and stops if an abnormality is detected. Separate chains require distinct chain IDs or TER records.
- This continuity check tests the plausibility of atom naming and distances; it does not replace validation against the full MD topology. SAS calculation does not require all bond orders.
- The analysis uses the exported, isolated polymer conformation. It does not include occlusion by periodic images or by the water and ions excluded from the analysis.
- Default Bondi-style radii for explicit atoms are H = 1.20, C = 1.70, O = 1.52, and S = 1.80 Å. Override them with `--radii radii.json`. MD Lennard-Jones sigma values are not automatically treated as van der Waals radii.
- Inputs without explicit hydrogens produce a warning because the default radii are not united-atom parameters. Use an atomic representation consistent with the hydrogen-containing examples.
- The default limit is 40 million grid points. The float32 distance field alone requires approximately 160 MB; peak memory also includes voxel-connectivity analysis and mesh arrays. Adjust `--max-grid-cells` for larger inputs.
- The script does not automatically assign files to RHP1–RHP5. A result from one conformation does not represent a time average, patch lifetime, or population average for an RHP composition.

## Upstream source code reviewed and design choices

**MolPatch**, commit `be919dca3bd35c6d3be1000e30de4e4bf98af625`.

- [AtomBased/ProteinPatch.py](https://github.com/ibivu/hydrophobic_patches/blob/be919dca3bd35c6d3be1000e30de4e4bf98af625/src/molpatch/AtomBased/ProteinPatch.py) constructs a graph from MSMS SES point clouds, atom labels, and distances between points.
- The `size()` method in [AtomBased/Patch.py](https://github.com/ibivu/hydrophobic_patches/blob/be919dca3bd35c6d3be1000e30de4e4bf98af625/src/molpatch/AtomBased/Patch.py) sums the full FreeSASA areas of atoms associated with a patch. It therefore does not directly integrate the connected surface region. When an atom's surface is divided among several regions, summing its full area is unsuitable for the area partitioning required here.
- The source includes protein-specific processing and older NetworkX calls, so the original scripts were not used directly.

**PEP-Patch**, commit `162a7c822acdedc9f281a7cc68efe88d8e33928f`.

- [patches.py](https://github.com/liedllab/surface_analyses/blob/162a7c822acdedc9f281a7cc68efe88d8e33928f/surface_analyses/patches.py) identifies connected components of selected vertices along triangular mesh edges.
- The `vertex_areas()` method in [surface.py](https://github.com/liedllab/surface_analyses/blob/162a7c822acdedc9f281a7cc68efe88d8e33928f/surface_analyses/surface.py) allocates one-third of each triangle's area to each vertex. The file also provides a distance-field approach to constructing SAS.
- [hydrophobic_potential.py](https://github.com/liedllab/surface_analyses/blob/162a7c822acdedc9f281a7cc68efe88d8e33928f/surface_analyses/hydrophobic_potential.py) provides a local hydrophobic-potential analysis based on contributions from surrounding atoms. The initial RHP implementation uses only EHMA geometric labels, without introducing an uncalibrated RHP hydrophobicity score.

This script is an independent implementation. It requires only NumPy, SciPy, and scikit-image; the complete upstream projects, MSMS, FreeSASA, MDTraj, and RDKit are not required. Mesh-edge connectivity and vertex-area allocation follow the algorithmic approach used in PEP-Patch.

## References

1. van Gils et al. *How sticky are our proteins? Quantifying hydrophobicity of the human proteome.* Bioinformatics Advances (2022). [doi:10.1093/bioadv/vbac002](https://doi.org/10.1093/bioadv/vbac002)
2. Hoerschinger et al. *PEP-Patch: Electrostatics in Protein–Protein Recognition, Specificity, and Antibody Developability.* JCIM (2023). [doi:10.1021/acs.jcim.3c01490](https://doi.org/10.1021/acs.jcim.3c01490)

Initial delivery status: source-code review, script development, and static syntax review were complete. **At that stage, the script had not been run on the supplied PDBs, and surface areas, performance, and mesh convergence had not been validated.**
