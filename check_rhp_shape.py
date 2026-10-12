#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量筛查 RHP PDB 的片状分布、低球度和批次离群值。

安装依赖: python -m pip install numpy matplotlib
运行示例: python check_rhp_shape.py "D:/RHP_MD" -o shape_check
          python check_rhp_shape.py . --recursive --sphericity-cutoff 0.30

默认读取每个 PDB 的第一个模型，排除氢/氘、水和常见游离离子。
每个文件应只包含一条完整 RHP；可用 --resnames 指定单体残基名。
各原子等权，不按原子质量加权，所有长度单位为 Angstrom (A)。

回转张量特征值按 l_min <= l_mid <= l_max 排序。主指标 sphericity
= sqrt(l_min/l_max)，即最短/最长主轴的 RMS 尺度比，范围 0--1。
这只是二阶矩的球度代理，不是基于表面积/体积的几何球度。
flatness = sqrt(l_min/l_mid)；inplane_ratio = sqrt(l_mid/l_max)。
片状分布的 flatness 很低、inplane_ratio 较高；细长分布则后者较低。
补充输出标准 relative shape anisotropy (kappa2) 及 Rg。

筛查阈值是经验起点，不是 MD 失败判据。批次 MAD 离群检测只适合
相似组成、链长和模拟条件的样本；混合批次请分文件夹分别运行。
不修改 PDB、不自动解周期性边界。导出前应按拓扑保持分子完整。
对上传示例中的已知单体名，附加检查相邻单体 C3--C2 距离。
这个检查不覆盖所有键，不能代替拓扑或轨迹验证。
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无图形界面的服务器也能保存图片。
import matplotlib.pyplot as plt

RHP_RESNAMES = {"HO1", "EH1", "EH2", "OE1", "OE2", "MM1", "MM2", "SP1", "SP2", "TEH"}
SOLVENT_IONS = {"HOH", "WAT", "SOL", "TIP3", "TIP", "NA", "NA+", "SOD", "CL", "CL-",
                "CLA", "K", "K+", "POT", "CA", "CA2", "MG", "MG2", "ZN", "LI", "CS"}
FIELDS = ["file", "status", "n_atoms", "n_residues", "residue_names", "n_models",
          "sphericity", "flatness", "inplane_ratio", "kappa2", "Rg_A",
          "rms_short_A", "rms_middle_A", "rms_long_A", "lambda_min_A2",
          "lambda_mid_A2", "lambda_max_A2", "sphericity_robust_z", "Rg_robust_z",
          "shape_hint", "backbone_links_checked", "backbone_max_link_A",
          "flagged", "reasons", "notes", "error"]


def read_pdb(path, include_hydrogens=False, resnames=None):
    """按 PDB 固定列读取，保留模型 1；坐标缺损不会被静默忽略。"""
    xyz, residues = [], {}
    models, in_first, segment = 0, True, 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        for lineno, line in enumerate(handle, 1):
            record = line[:6].strip()
            if record == "MODEL":
                models += 1
                in_first = models == 1
                continue
            if record == "ENDMDL":
                in_first = False
                continue
            if not in_first:
                continue
            if record == "TER":
                segment += 1
                continue
            if record not in ("ATOM", "HETATM"):
                continue
            resname = line[17:20].strip()
            if resnames is not None:
                if resname not in resnames:
                    continue
            elif resname.upper() in SOLVENT_IONS:
                continue
            # MD 文件一般无 alternate locations；只取空白或 A，避免重复。
            if line[16:17] not in (" ", "A", ""):
                continue
            atomname = line[12:16].strip()
            element = line[76:78].strip().upper()
            if not element:
                element = re.sub(r"^[0-9]+", "", atomname).upper()[:1]
            if not include_hydrogens and element in ("H", "D", "T"):
                continue
            try:
                point = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            except ValueError as exc:
                raise ValueError(f"invalid coordinates at line {lineno}") from exc
            if not np.isfinite(point).all():
                raise ValueError(f"non-finite coordinates at line {lineno}")
            xyz.append(point)
            key = (segment, line[21:22], line[22:26], line[26:27])
            residues.setdefault(key, {"name": resname, "atoms": {}})["atoms"][atomname] = np.array(point)
    if len(xyz) < 3:
        raise ValueError(f"fewer than 3 selected atoms ({len(xyz)})")
    names = sorted({r["name"] for r in residues.values()})
    meta = {"n_atoms": len(xyz), "n_residues": len(residues),
            "residue_names": ",".join(names), "n_models": max(1, models)}
    notes = []
    if models > 1:
        notes.append("first_model_only")
    if len({(k[0], k[1]) for k in residues}) > 1:
        notes.append("multiple_segments_or_chains: metric describes their combined distribution")
    # 此处 C3(i)--C2(i+1) 只针对上传文件中的单体命名，不推断未知拓扑。
    links = []
    items = list(residues.items())
    for (ka, ra), (kb, rb) in zip(items, items[1:]):
        if ka[:2] != kb[:2] or ra["name"] not in RHP_RESNAMES or rb["name"] not in RHP_RESNAMES:
            continue
        try:
            consecutive = int(kb[2]) == int(ka[2]) + 1 and ka[3].strip() == kb[3].strip() == ""
        except ValueError:
            consecutive = False
        if consecutive and "C3" in ra["atoms"] and "C2" in rb["atoms"]:
            links.append(np.linalg.norm(ra["atoms"]["C3"] - rb["atoms"]["C2"]))
    meta.update(backbone_links_checked=len(links),
                backbone_max_link_A=float(max(links)) if links else "", notes="; ".join(notes))
    return np.asarray(xyz), meta


def shape_metrics(xyz):
    """平移和旋转不改变结果；特征向量用于绘制主轴投影。"""
    centered = xyz - xyz.mean(axis=0)
    tensor = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    small, middle, large = eigenvalues
    trace = float(eigenvalues.sum())
    if trace <= 0:
        raise ValueError("all selected coordinates coincide")
    kappa2 = 1 - 3 * (small * middle + small * large + middle * large) / trace ** 2
    rms = np.sqrt(eigenvalues)
    metrics = {"sphericity": float(rms[0] / rms[2]),
               "flatness": float(rms[0] / rms[1]) if rms[1] > 0 else 0.0,
               "inplane_ratio": float(rms[1] / rms[2]),
               "kappa2": float(np.clip(kappa2, 0, 1)), "Rg_A": float(np.sqrt(trace)),
               "rms_short_A": float(rms[0]), "rms_middle_A": float(rms[1]),
               "rms_long_A": float(rms[2]), "lambda_min_A2": float(small),
               "lambda_mid_A2": float(middle), "lambda_max_A2": float(large)}
    return metrics, centered @ eigenvectors[:, ::-1]


def robust_z(values):
    """MAD 标准化；样本少于 10 或 MAD 近零时停用，不制造伪离群。"""
    values = np.asarray(values, dtype=float)
    if len(values) < 10:
        return np.full(len(values), np.nan)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad <= 1e-12 * max(1.0, abs(median)):
        return np.full(len(values), np.nan)
    return 0.6744897501960817 * (values - median) / mad


def screen(rows, args):
    valid = [r for r in rows if r["status"] == "ok"]
    zs = robust_z([r["sphericity"] for r in valid])
    zr = robust_z([r["Rg_A"] for r in valid])
    for r, s_z, rg_z in zip(valid, zs, zr):
        reasons = []
        r["sphericity_robust_z"] = float(s_z) if np.isfinite(s_z) else ""
        r["Rg_robust_z"] = float(rg_z) if np.isfinite(rg_z) else ""
        if r["flatness"] < args.flatness_cutoff and r["inplane_ratio"] >= 0.5:
            r["shape_hint"] = "sheet_like"
            reasons.append("sheet_like")
        elif r["inplane_ratio"] < 0.35:
            r["shape_hint"] = "rod_like"
        else:
            r["shape_hint"] = "3D_or_intermediate"
        if r["sphericity"] < args.sphericity_cutoff:
            reasons.append("low_sphericity")
        if s_z < -args.z_cutoff:
            reasons.append("sphericity_low_outlier")
        if abs(rg_z) > args.z_cutoff:
            reasons.append("Rg_outlier")
        if r["backbone_max_link_A"] != "" and r["backbone_max_link_A"] > args.link_cutoff:
            reasons.append("backbone_gap_possible_PBC_or_bad_geometry")
        if "multiple_segments_or_chains" in r.get("notes", ""):
            reasons.append("multiple_segments_or_chains_review")
        r["flagged"] = bool(reasons)
        r["reasons"] = "; ".join(reasons)
    return valid


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_summary(valid, output, args):
    flagged = np.array([r["flagged"] for r in valid])
    colors = np.where(flagged, "#D55346", "#337FA1")
    s = np.array([r["sphericity"] for r in valid])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.scatter(np.arange(1, len(valid) + 1), s, c=colors, s=18, alpha=0.8)
    ax.axhline(args.sphericity_cutoff, color="#D55346", ls="--", label="screening cutoff")
    ax.set(xlabel="File index (natural filename order)", ylabel="Sphericity: short / long RMS axis", ylim=(-0.02, 1.03))
    ax.legend(frameon=False, fontsize=8)
    ax = axes[0, 1]
    ax.hist(s, bins=np.linspace(0, 1, 31), color="#337FA1", edgecolor="white")
    ax.axvline(args.sphericity_cutoff, color="#D55346", ls="--")
    ax.set(xlabel="Sphericity", ylabel="Number of files", xlim=(0, 1))
    ax = axes[1, 0]
    ax.scatter([r["flatness"] for r in valid], [r["inplane_ratio"] for r in valid], c=colors, s=22, alpha=0.8)
    ax.axvline(args.flatness_cutoff, color="0.5", ls="--")
    ax.axhline(0.5, color="0.5", ls="--")
    ax.text(0.02, 0.98, "sheet-like corner", va="top", fontsize=8, transform=ax.transAxes)
    ax.text(0.98, 0.98, "isotropic", ha="right", va="top", fontsize=8, transform=ax.transAxes)
    ax.set(xlabel="Flatness: short / middle RMS axis", ylabel="In-plane ratio: middle / long RMS axis", xlim=(-0.02, 1.03), ylim=(-0.02, 1.03))
    ax = axes[1, 1]
    ax.scatter([r["Rg_A"] for r in valid], [r["kappa2"] for r in valid], c=colors, s=22, alpha=0.8)
    ax.set(xlabel="Geometric radius of gyration (A)", ylabel="Relative shape anisotropy (kappa squared)")
    # 仅标记最多 8 个最低球度候选，避免 600 个名称挤满图。
    for i in sorted(np.flatnonzero(flagged), key=lambda k: s[k])[:8]:
        axes[0, 0].annotate(valid[i]["file"], (i + 1, s[i]), xytext=(3, 4), textcoords="offset points", fontsize=6)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"RHP shape screening | Samples: {len(valid)} | Review candidates: {flagged.sum()}\nRed = flagged; blue = no screening flag", fontsize=13)
    fig.savefig(output / "shape_summary.png", dpi=180)
    plt.close(fig)


def make_previews(valid, root, output, args):
    if args.preview_count == 0:
        return
    ordered = sorted(valid, key=lambda r: r["sphericity"])
    selected = ordered[:args.preview_count]
    median = ordered[len(ordered) // 2]
    if median not in selected:
        selected.append(median)
    fig, axes = plt.subplots(len(selected), 3, figsize=(10, 3.15 * len(selected)), squeeze=False, constrained_layout=True)
    for row_axes, r in zip(axes, selected):
        xyz, _ = read_pdb(root / r["file"], args.include_hydrogens, args.resnames)
        _, projected = shape_metrics(xyz)
        limit = max(float(np.abs(projected).max()) * 1.05, 1.0)
        color = "#D55346" if r["flagged"] else "#337FA1"
        for ax, (i, j) in zip(row_axes, ((0, 1), (0, 2), (1, 2))):
            ax.scatter(projected[:, i], projected[:, j], s=2, alpha=0.35, color=color, rasterized=True)
            ax.set(xlim=(-limit, limit), ylim=(-limit, limit),
                   xlabel=f"Principal axis {i+1} (A)", ylabel=f"Principal axis {j+1} (A)")
            ax.set_aspect("equal")
        row_axes[0].set_title(f"{r['file']}\nS={r['sphericity']:.3f}, kappa2={r['kappa2']:.3f}", fontsize=9)
        row_axes[1].set_title(r["shape_hint"].replace("_", " "), fontsize=9)
        row_axes[2].set_title("Flagged for review" if r["flagged"] else "No screening flag", fontsize=9)
    fig.suptitle("Lowest sphericity samples + median sample\nThree orthogonal principal-axis views; equal scales within each row", fontsize=12)
    fig.savefig(output / "shape_previews.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Screen RHP PDB files for sheet-like or unusual atom distributions.")
    parser.add_argument("folder", nargs="?", default=".", type=Path, help="folder containing PDB files (default: current folder)")
    parser.add_argument("-o", "--output", type=Path, default=Path("shape_check"))
    parser.add_argument("--recursive", action="store_true", help="also search subfolders")
    parser.add_argument("--pattern", default="*.pdb", help="filename glob, e.g. seq*.pdb")
    parser.add_argument("--include-hydrogens", action="store_true")
    parser.add_argument("--resnames", nargs="+", help="select only these residue names, e.g. HO1 EH1 EH2 OE1 OE2 MM1 MM2 SP1 SP2 TEH")
    parser.add_argument("--sphericity-cutoff", type=float, default=0.30, help="flag S below this value (default: 0.30)")
    parser.add_argument("--flatness-cutoff", type=float, default=0.25, help="sheet-like: flatness below this and inplane_ratio >= 0.5")
    parser.add_argument("--z-cutoff", type=float, default=3.5, help="MAD outlier threshold; requires >=10 valid files")
    parser.add_argument("--link-cutoff", type=float, default=3.0, help="flag known RHP C3--C2 links above this length in A")
    parser.add_argument("--preview-count", type=int, default=6, help="number of lowest-S samples to plot, plus median; 0 disables")
    args = parser.parse_args(argv)
    if not (0 < args.sphericity_cutoff < 1 and 0 < args.flatness_cutoff < 1):
        parser.error("shape cutoffs must be between 0 and 1")
    if args.z_cutoff <= 0 or args.link_cutoff <= 0 or args.preview_count < 0:
        parser.error("z/link cutoffs must be positive; preview-count must be nonnegative")
    if not args.folder.is_dir():
        parser.error(f"not a folder: {args.folder}")
    args.resnames = set(args.resnames) if args.resnames else None
    root = args.folder.resolve()
    paths = root.rglob(args.pattern) if args.recursive else root.glob(args.pattern)
    def natural_key(path):
        return [int(p) if p.isdigit() else p.casefold() for p in re.split(r"(\d+)", str(path.relative_to(root)))]
    paths = sorted((p for p in paths if p.is_file()), key=natural_key)
    if not paths:
        parser.error(f"no files matching {args.pattern!r} in {root}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, path in enumerate(paths, 1):
        row = {"file": path.relative_to(root).as_posix(), "status": "ok", "error": ""}
        try:
            xyz, meta = read_pdb(path, args.include_hydrogens, args.resnames)
            metrics, _ = shape_metrics(xyz)
            row.update(meta)
            row.update(metrics)
        except (OSError, ValueError, np.linalg.LinAlgError) as exc:
            row.update(status="error", flagged=True, reasons="read_or_coordinate_error", error=str(exc))
        rows.append(row)
        if index % 50 == 0 or index == len(paths):
            print(f"Processed {index}/{len(paths)} files", flush=True)
    valid = screen(rows, args)
    candidates = sorted((r for r in rows if r["flagged"]), key=lambda r: (r["status"] == "ok", r.get("sphericity", -1)))
    write_csv(args.output / "shape_metrics.csv", rows)
    write_csv(args.output / "review_candidates.csv", candidates)
    if valid:
        make_summary(valid, args.output, args)
        make_previews(valid, root, args.output, args)
    print(f"\nValid: {len(valid)}; unreadable/invalid: {len(rows)-len(valid)}; review candidates: {len(candidates)}")
    if valid:
        worst = min(valid, key=lambda r: r["sphericity"])
        print(f"Lowest sphericity: {worst['file']} (S={worst['sphericity']:.4f}, kappa2={worst['kappa2']:.4f})")
    for r in candidates[:15]:
        print(f"  REVIEW {r['file']}: {r['reasons']}")
    print(f"Results: {args.output.resolve()}")
    print("Flags identify review candidates, not confirmed MD failures. Analyze matched compositions/chain lengths together.")
    print("Coordinates are not unwrapped. Export whole molecules using the simulation topology before interpreting shape.")
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
