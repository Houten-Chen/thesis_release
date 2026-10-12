# RHP 连续表面与三角网格对比

`render_rhp_patch_mesh.py` 是可独立运行的新脚本。分析部分保留现有 `rhp_surface_patches.py` 的算法，渲染配色沿用现有 mosaic renderer。

## 运行

```bash
python -m pip install numpy scipy scikit-image matplotlib pillow vtk
python render_rhp_patch_mesh.py seq151_1.pdb --out comparison
```

同时处理多个 PDB：

```bash
python render_rhp_patch_mesh.py seq151_1.pdb seq658_3.pdb --out comparison
```

在桌面上旋转、确认角度后再输出：

```bash
python render_rhp_patch_mesh.py seq151_1.pdb --interactive --export-vtp
```

两个视图共用相机。鼠标拖动旋转，滚轮缩放，Enter 保存，Esc 跳过。每个 PDB 独立预览。

默认网格间距 0.35 Å，探针半径 1.4 Å。可用 `--grid-spacing`、`--probe-radius` 修改；应与 thesis 中实际分析参数一致。`--elev 25 --azim -55` 设置初始观察角度。`--detail-span 9` 设置局部放大区域的高度，单位 Å；`--detail-center X Y Z` 可指定区域中心。

## 图中对应的计算

1. 全部聚合物原子及显式氢构建未平滑的外部 SAS，即范德华半径加探针半径后球体并集的边界。
2. 每个网格顶点归属使“顶点至原子中心距离减扩张半径”最小的原子。
3. 每个三角形面积均分给三个顶点。染色时将原三角形共面细分为六个小三角形，每个原顶点拥有其中两个，准确表达其三分之一面积。
4. EHMA alkyl 选择 C7–C14 及相连氢。沿原网格边连接已选顶点，连通分量构成 patch。patch 面积汇总这些顶点的面积贡献。

图中深色线是原始三角网格边。局部放大图中的浅色线标出三个顶点各自的等面积区域。共面细分只用于显示，patch 连接关系始终取自原始网格。没有几何平滑或 mesh decimation。

`*_comparison.png` 并排比较同一表面的连续显示和三角网格显示，颜色代表整个 monomer。`*_EHMA_patches.png` 左侧仅突出 EHMA 烷基表面，右侧以不同颜色区分连通 patch；其余部分为浅灰色。后者更直接对应 patch count 的选择规则。

## 输出

每个 PDB 输出上述两幅带局部放大的对比图，各视图的独立 PNG，以及：

- `*_method.json` 保存参数、相机、网格大小、统计值和面积守恒检查。
- `*_EHMA_patches.csv` 保存各 patch 面积、顶点数及对应 residue。
- `*_surface_mesh.npz` 保存用于分析的原始网格和顶点标签。
- 加 `--export-vtp` 时，另输出连续连接、按面染色的 `*_colored_surface.vtp`，可用 ParaView 等软件读取。

脚本会核对每个原三角形的面积守恒、每个顶点的面积分配和每个 patch 的染色面积。示例构象用于解释方法；各 RHP 的统计分布应使用完整数据集。

无桌面的 Linux 使用 VTK 9.5 及以上的 EGL 渲染。也可添加 `--renderer matplotlib`，但大网格运行更慢。Windows/macOS 通常可直接使用 VTK 桌面渲染。
