"""Headless PNG/PDF output with configurable labels, colors, and plot limits."""
from pathlib import Path
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

from .common import (check_embedding_metadata, load_pca, meta_path, prepare_outputs,
                     project, read_latents, save_json, sha256)


def use_font(name="Arial"):
    available = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = name if name in available else "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


def save_figure(fig, stem, overwrite=False, dpi=300):
    stem = Path(stem)
    paths = [Path(str(stem) + ".png"), Path(str(stem) + ".pdf")]
    prepare_outputs(paths, overwrite)
    fig.savefig(paths[0], dpi=dpi, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)


def heatmap(frame, grid, column, stem, label, limits=None, font="Arial", overwrite=False, xlim=None, ylim=None, cmap="RdBu_r"):
    use_font(font)
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    kwargs = {} if limits is None else {"vmin": limits[0], "vmax": limits[1]}
    image = ax.imshow(frame[column].to_numpy().reshape(grid.ny, grid.nx), origin="lower", extent=grid.extent,
                      cmap=cmap, aspect="auto", **kwargs)
    ax.set(xlabel="PC1", ylabel="PC2")
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    ax.spines[["top", "right"]].set_visible(False)
    fig.colorbar(image, ax=ax, label=label)
    fig.tight_layout()
    save_figure(fig, stem, overwrite)


def load_groups(path):
    path = Path(path)
    config = json.loads(path.read_text())
    groups = config.get("groups") if isinstance(config, dict) else config
    if not isinstance(groups, list) or not groups:
        raise ValueError("Group JSON needs a nonempty groups array")
    seen = set()
    output = []
    for original in groups:
        group = dict(original)
        if not {"label", "file", "kind"} <= set(group) or group["kind"] not in {"protein", "rhp"}:
            raise ValueError("Every group requires label, file, and kind=protein/rhp")
        if group["label"] in seen:
            raise ValueError("Group labels must be unique")
        seen.add(group["label"])
        group["file"] = str((path.parent / group["file"]).resolve())
        output.append(group)
    return output


def plot_kde(groups_path, pca_path, output_dir, threshold=.04, font="Arial", overwrite=False, xlim=None, ylim=None):
    if not 0 <= threshold < 1:
        raise ValueError("KDE threshold must lie in [0, 1)")
    groups = [g for g in load_groups(groups_path) if g.get("plot", True)]
    if not groups:
        raise ValueError("No groups are enabled for plotting")
    pca = load_pca(pca_path)
    check_embedding_metadata([g["file"] for g in groups], pca=pca)
    output_dir = Path(output_dir)
    csv_path = output_dir / "projected_windows.csv"
    prepare_outputs([csv_path, meta_path(csv_path), output_dir / "sequence_space.png", output_dir / "sequence_space.pdf",
                     output_dir / "sequence_space_legend.png", output_dir / "sequence_space_legend.pdf"], overwrite,
                    [g["file"] for g in groups] + [groups_path, pca_path])
    use_font(font)
    frames = []
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    handles = []
    for group in groups:
        vectors, rows = read_latents(group["file"])
        coords = project(vectors, pca)
        data = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "group": group["label"],
                             "kind": group["kind"], "source_row": [r[0] for r in rows],
                             "window_index": [r[1] for r in rows]})
        frames.append(data)
        color = group.get("color", "#777777")
        display = group.get("display_label", group["label"])
        filled = group.get("fill", group["kind"] == "protein" and not group.get("reference_outline", False))
        bw = group.get("bw_adjust", 1.35 if group["label"] == "Disordered region" else 1)
        if len(coords) < 3 or np.linalg.matrix_rank(np.cov(coords.T)) < 2:
            warnings.warn(f"{display}: too few independent points for KDE; displaying points")
            ax.scatter(coords[:, 0], coords[:, 1], color=color, s=5)
        else:
            if filled:
                sns.kdeplot(data=data, x="PC1", y="PC2", fill=True, thresh=threshold, levels=4,
                            cmap=sns.light_palette(color, as_cmap=True), alpha=group.get("alpha", .7), bw_adjust=bw, ax=ax)
            sns.kdeplot(data=data, x="PC1", y="PC2", fill=False, thresh=threshold,
                        levels=4 if group["kind"] == "rhp" else 1, color=color, bw_adjust=bw,
                        alpha=.7, linewidths=.8 if group["kind"] == "rhp" else 2, ax=ax)
        handles.append(Patch(facecolor=color if filled else "none", edgecolor=color, label=display))
    ax.set(xlabel="PC1", ylabel="PC2")
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, output_dir / "sequence_space", overwrite)
    legend, legend_ax = plt.subplots(figsize=(5, max(1.5, .3 * len(handles))))
    legend_ax.axis("off")
    legend_ax.legend(handles=handles, loc="center", frameon=False)
    save_figure(legend, output_dir / "sequence_space_legend", overwrite)
    pd.concat(frames, ignore_index=True).to_csv(csv_path, index=False)
    save_json(meta_path(csv_path), {"pca_sha256": sha256(pca_path), "groups": groups, "threshold": threshold,
                                  "sampling": "all valid complete-window embeddings; no resampling"})


def rank_curves(points, output_dir, font="Arial", overwrite=False):
    use_font(font)
    for label, frame in points.groupby("RHP", sort=False):
        values = frame.structure_tendency.to_numpy()
        finite = np.isfinite(values)
        ranked = np.r_[np.sort(values[finite])[::-1], np.full((~finite).sum(), np.nan)]
        fig, ax = plt.subplots(figsize=(3.5, 3.2))
        ax.plot(np.arange(len(ranked)), ranked, color="#007ACC", lw=1.5)
        ax.set(xlabel=f"{label} window rank", ylabel="Protein-derived structure tendency")
        ax.spines[["top", "right"]].set_visible(False)
        save_figure(fig, Path(output_dir) / (label + "_rank"), overwrite)
