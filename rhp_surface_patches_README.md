# 单个 RHP PDB 的表面组成与 EHMA patch 分析

脚本 `rhp_surface_patches.py` 接收一个 PDB。已经根据两个公开仓库的核心源码和上传 PDB 的命名方式设计，尚未运行 PDB 分析，也没有完成网格收敛验证。

## 安装与使用

建议使用 Python 3.10 或更高版本。

```bash
python -m pip install numpy scipy scikit-image
python rhp_surface_patches.py seq658_1.pdb --out seq658_1_surface --grid-spacing 0.5 --export-ply
```

文件路径有空格时，请用双引号包围路径。`--export-ply` 可省略。默认输出目录是 PDB 所在目录中的 `<文件名>_surface`。已有输出默认不会覆盖；覆盖时显式加 `--overwrite`。

## 这份脚本具体计算什么

1. 使用全部聚合物原子（包括文件中已有的显式氢）构建表面。水和常见离子被忽略。
2. 表面采用 **SAS，溶剂可及表面**。它是原子范德华半径加探针半径后，所有球体并集的边界。默认探针半径为 1.4 Å。
3. 使用三角网格表达该表面。默认排除探针中心无法从外界到达的封闭内腔；开放沟槽和孔道保留。外界连通性由体素网格判断，同样受网格间距影响。
4. 每个顶点归属使 `顶点到原子中心的距离 - 扩张半径` 最小的原子。这样可以考虑不同元素半径，避免只按最近原子中心分配。
5. 每个三角形面积的三分之一分配给各顶点，再按原子和 monomer 汇总。因此四类 monomer 的面积之和等于总表面积。
6. 将来自 EHMA 烷基碳及其氢的顶点标为 EHMA alkyl。沿网格边连接这些顶点，每个连通区域就是一个 patch。没有另外使用空间距离 cutoff。
7. 每个 patch 的面积为其顶点分配面积之和。所有 EHMA patch 面积之和等于 EHMA 烷基暴露面积。混合标签的边界三角形按顶点占比划分，边界近似会随网格加密而改变。

这里的 patch 是 **EHMA 烷基贡献的连续几何暴露区域**。它不包含通过 MMA 等其他 monomer 连接的混合疏水 patch，也没有用水合自由能或 logP 势场定义疏水性。

**SAS 与 SES（溶剂排除表面）不同**。本脚本的面积应与同一表面定义下的结果比较。本文献分析不是对原始 MolPatch 或 PEP-Patch 数值的复现。

## 已匹配的 PDB 命名

| Monomer | residue 名称 |
|---|---|
| MMA | MM1、MM2、HM1、HM2、TMM |
| EHMA | EH1、EH2、HE1、HE2、TEH |
| OEGMA | OE1、OE2、HO1、HO2、TOE |
| SPMA | SP1、SP2、HS1、HS2、TSP |

上传文件中实际出现的 MM1/MM2/HM2/TMM、EH1/EH2/TEH、OE1/OE2/HO1/TOE、SP1/SP2 已检查其重原子命名。其他首末端别名是可配置的候选；如使用，需确认化学结构对应关系。首末端 residue 计入其所属 monomer 的 repeat-unit 数量。

EHMA 烷基选择为 `C7,C8,C9,C10,C11,C12,C13,C14`。氢原子归到同一 residue 内最近的重原子，只有归到上述碳的氢才进入 EHMA patch。`C6/O1/O2` 是酯基，不进入烷基 patch。

`backbone_common` 汇总 C1–C5 及相连氢，包含聚合物主链、共同的甲基和端部碳，并非严格意义的纯主链面积。`ester` 为 C6/O1/O2，`sidechain` 为其余基团。monomer 表面占比使用整个 repeat unit 的面积；EHMA patch 使用烷基侧链的面积。

如 residue 名称不同，可另建 JSON，例如：

```json
{"MYE": "EHMA", "MYO": "OEGMA"}
```

再加 `--residue-map mapping.json`。如果 EHMA 碳原子名称不同，使用 `--ehma-carbons C7,C8,C9,C10,C11,C12,C13,C14` 指定实际名称。未知 residue 会报错，避免静默误分类。

## 输出文件

| 文件 | 内容 |
|---|---|
| `summary.json` | 参数、表面定义、原子数、网格信息、全部 patch 统计及警告 |
| `summary.csv` | 便于查看的单行数值汇总 |
| `monomer_surface.csv` | 四类 monomer 的实际数量比例、表面面积比例和基团面积 |
| `EHMA_patches.csv` | 每个 patch 的面积、对应 EHMA residue、面积加权中心，按面积从大到小排序 |
| `residue_surface.csv` | 每个 repeat unit 的暴露面积和 EHMA 烷基暴露面积 |
| `atom_surface.csv` | 每个原子的面积、标签、氢的归属重原子，便于核对选择规则 |
| `surface_mesh.npz` | 三角网格、顶点面积、来源原子和 EHMA patch ID；坐标单位为 Å |
| `surface_monomers.ply` | 可选；按四类 monomer 着色的网格 |
| `surface_EHMA_patches.ply` | 可选；各 EHMA patch 着色，其余表面为灰色 |

PLY 可用支持顶点颜色的 MeshLab 等软件查看。CSV 使用 UTF-8 BOM 编码，便于 Excel 打开。此脚本仅处理指定的一个 MODEL，默认是 PDB 中第一个 MODEL；使用 `--model 2` 选择第二个。

## 平均 patch 面积

令各 patch 面积为 A_j。

- `EHMA_patch_mean_all_A2` 是普通平均，即 `sum(A_j) / patch_count`。
- `EHMA_patch_area_weighted_mean_A2` 是面积加权平均，即 `sum(A_j**2) / sum(A_j)`。在 EHMA 暴露表面均匀随机取点时，这就是该点所在 patch 的平均面积。
- `EHMA_largest_patch_A2` 是最大 patch 面积。
- 另保存面积达到 `--min-patch-area` 的 patch 的数量和普通平均，默认阈值为 10 Å²。这是便于检查小碎片影响的**可调分析阈值**，没有在 RHP 上验证为物理阈值。原始全部 patch 始终保留。

没有暴露 patch 时，普通平均和面积加权平均在 JSON 中为 null，在 CSV 中为空；最大面积和 patch 数量为零。

所有面积统计单位为 Å²；100 Å² = 1 nm²。表面面积百分比与 monomer 数量百分比含义不同，尤其 OEGMA 侧链尺寸较大。不要把二者的比值直接解释为已排除尺寸效应的表面富集程度。

## 本地怎样检查 resolution

这份方案使用表面网格，因此需要检查网格间距。先对**同一个 PDB、同一套原子半径和同一个探针半径**分别运行：

```bash
python rhp_surface_patches.py seq658_1.pdb --out scan_050 --grid-spacing 0.50
python rhp_surface_patches.py seq658_1.pdb --out scan_035 --grid-spacing 0.35
python rhp_surface_patches.py seq658_1.pdb --out scan_025 --grid-spacing 0.25
```

比较总面积、各类 monomer 表面百分比、EHMA 烷基面积、最大 patch、面积加权平均，以及全部 patch 和面积过滤后的数量与普通平均。

如果总面积稳定、patch 数量或普通平均仍变化明显，往往需要继续检查化学边界分辨率和细小连接通道。更细的网格会提高内存和运行时间，不能预先保证哪个间距足够。初始默认 0.5 Å 只是本地试运行的起点，并非经过验证的最佳分辨率。

`--probe-radius` 改变几何可及尺度，会改变物理定义；它与用于数值收敛的 `--grid-spacing` 是两个不同参数。

## 坐标和参数限制

- 输入应是完整、连续的聚合物坐标。脚本不会自动重建周期性边界 image。它检查 residue 内的共价距离连通性，以及相邻 residue 的 C3–C2 主链连接；异常时停止。不同链需有独立 chain ID 或 TER 分隔。
- 此连续性检查是命名和距离的合理性检查，不能替代完整 MD topology 的验证。SAS 计算不需要知道全部键级。
- 分析的是已导出的孤立聚合物构象，没有加入周期性相邻 image 的遮挡，也没有加入已忽略水和离子的遮挡。
- 默认使用显式原子的 Bondi 风格半径 H=1.20、C=1.70、O=1.52、S=1.80 Å。可用 `--radii radii.json` 覆盖。这里没有自动把 MD 的 Lennard-Jones sigma 当作范德华半径。
- 无显式氢的输入会给出警告，因为默认半径并非 united-atom 参数。应与这些带氢示例保持一致的原子模型。
- 默认最多 4,000 万网格点。仅 float32 距离场就占约 160 MB，实际峰值还包括体素连通分析和网格数组。更大输入可调整 `--max-grid-cells`。
- 本脚本没有将文件自动归到 RHP1–RHP5，也不把一个构象的结果解释为时间平均、patch 寿命或一类 RHP 的 population 平均。

## 检查过的上游源码与设计取舍

**MolPatch**，commit `be919dca3bd35c6d3be1000e30de4e4bf98af625`。

- [AtomBased/ProteinPatch.py](https://github.com/ibivu/hydrophobic_patches/blob/be919dca3bd35c6d3be1000e30de4e4bf98af625/src/molpatch/AtomBased/ProteinPatch.py) 根据 MSMS 的 SES 点云、原子标签和点间距离构图。
- [AtomBased/Patch.py](https://github.com/ibivu/hydrophobic_patches/blob/be919dca3bd35c6d3be1000e30de4e4bf98af625/src/molpatch/AtomBased/Patch.py) 的 `size()` 汇总该 patch 涉及原子的整块 FreeSASA 面积。因此不是直接积分该连通表面区域；在原子表面被分到多个区域时，直接汇总整块原子面积不适合我们希望保留的面积分区。
- 源码包含蛋白质专用处理和旧版 NetworkX 调用，因此没有直接套用原脚本。

**PEP-Patch**，commit `162a7c822acdedc9f281a7cc68efe88d8e33928f`。

- [patches.py](https://github.com/liedllab/surface_analyses/blob/162a7c822acdedc9f281a7cc68efe88d8e33928f/surface_analyses/patches.py) 沿三角网格边寻找已选顶点的连通分量。
- [surface.py](https://github.com/liedllab/surface_analyses/blob/162a7c822acdedc9f281a7cc68efe88d8e33928f/surface_analyses/surface.py) 的 `vertex_areas()` 将每个三角形的面积均分给三个顶点，且提供距离场构建 SAS 的路线。
- [hydrophobic_potential.py](https://github.com/liedllab/surface_analyses/blob/162a7c822acdedc9f281a7cc68efe88d8e33928f/surface_analyses/hydrophobic_potential.py) 也提供周围原子贡献的局部疏水势分析。但初版仅做 EHMA 几何标签，不引入尚未校准的 RHP 疏水性评分。

本脚本采用独立实现，不需要安装这两个完整项目、MSMS、FreeSASA、MDTraj 或 RDKit。只需要 NumPy、SciPy、scikit-image。网格边连通和顶点面积分配沿用 PEP-Patch 的算法思路。

## 文献

1. van Gils et al. *How sticky are our proteins? Quantifying hydrophobicity of the human proteome.* Bioinformatics Advances (2022). https://doi.org/10.1093/bioadv/vbac002
2. Hoerschinger et al. *PEP-Patch: Electrostatics in Protein–Protein Recognition, Specificity, and Antibody Developability.* JCIM (2023). https://doi.org/10.1021/acs.jcim.3c01490

交付状态：只完成源码检查、脚本编写和静态语法审阅，**未运行本脚本，未对这些 PDB 计算表面积，未验证性能或网格收敛**。
