#!/usr/bin/env python3
"""Command-line entry points for the Chapter 3 release."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import math
from pathlib import Path
import tempfile

import pandas as pd

from ch3lib.common import (encode_sequence, input_records, meta_path, prepare_outputs,
                           read_table, save_json, sliding_positions)


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return value


def common_output(parser):
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs")


def table_arguments(parser):
    parser.add_argument("--input", nargs="+", required=True, type=Path, help="Explicit CSV/Excel input paths")
    parser.add_argument("--sequence-column", default="Sequence")


def plot_arguments(parser):
    parser.add_argument("--font", default="Arial", help="Arial when installed; otherwise DejaVu Sans")
    parser.add_argument("--xlim", nargs=2, type=float)
    parser.add_argument("--ylim", nargs=2, type=float)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    windows = commands.add_parser("windows", help="Encode complete protein windows for upstream training/input preparation")
    table_arguments(windows)
    windows.add_argument("--output", required=True, type=Path)
    windows.add_argument("--window", type=positive_int, default=100, help="Original conversion default is 100, not 50")
    windows.add_argument("--step", type=positive_int, default=15)
    windows.add_argument("--unknown-policy", choices=["error", "mma"], default="error")
    common_output(windows)

    imported = commands.add_parser("import-rhpapp", help="Convert headerless RHPapp monomer-array CSV exports; simulation remains upstream")
    imported.add_argument("--input", nargs="+", type=Path, required=True)
    imported.add_argument("--output", type=Path, required=True)
    imported.add_argument("--source-codes", nargs=4, required=True, metavar=("MMA_CODE", "OEGMA_CODE", "EHMA_CODE", "SPMA_CODE"),
                          help="Exported monomer IDs for MMA, OEGMA, EHMA, SPMA in that order; declare the actual upstream ordering")
    common_output(imported)

    embed = commands.add_parser("embed", help="Embed protein or RHP 50-mers with separately obtained AE_MP code/weights")
    table_arguments(embed)
    embed.add_argument("--kind", required=True, choices=["protein", "rhp"])
    embed.add_argument("--ae-repo", required=True, type=Path)
    embed.add_argument("--checkpoint", required=True, type=Path)
    embed.add_argument("--model-config", type=Path, help="Optional JSON override of the original AE_MP hidden sizes")
    embed.add_argument("--output-dir", required=True, type=Path)
    embed.add_argument("--step", type=positive_int, default=15)
    embed.add_argument("--batch-size", type=positive_int, default=1000)
    embed.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    embed.add_argument("--unknown-policy", choices=["error", "mma"], default="error")
    common_output(embed)

    q3 = commands.add_parser("predict-q3", help="Use ESM2 layer 33 and Porter6 predictor3; external weights required")
    table_arguments(q3)
    q3.add_argument("--esm-checkpoint", required=True, type=Path)
    q3.add_argument("--porter-source", required=True, type=Path, help="predictor3/model/cbrcnn_stage1.py")
    q3.add_argument("--porter-checkpoint", required=True, type=Path)
    q3.add_argument("--output-dir", required=True, type=Path)
    q3.add_argument("--chunk-size", type=positive_int, default=9500, help="Original default; reduce this for limited GPU memory")
    q3.add_argument("--device", default="auto")
    q3.add_argument("--on-error", choices=["raise", "record"], default="raise")
    common_output(q3)

    fit = commands.add_parser("fit-pca", help="Fit and save one explicitly selected reference; never implicit during analysis")
    source = fit.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", nargs="+", type=Path)
    source.add_argument("--groups", type=Path, help="Use only entries with fit_reference=true")
    fit.add_argument("--output", required=True, type=Path)
    fit.add_argument("--components", type=positive_int, default=5)
    common_output(fit)

    kde = commands.add_parser("plot-kde", help="Project named groups through a fixed reference and plot KDE contours")
    kde.add_argument("--groups", required=True, type=Path)
    kde.add_argument("--pca", required=True, type=Path)
    kde.add_argument("--output-dir", required=True, type=Path)
    kde.add_argument("--threshold", type=float, default=.04)
    plot_arguments(kde); common_output(kde)

    for name, default_min, explanation in [("lag-map", 10, "HLB lag profiles, variance of bin-mean profiles, and entropy"),
                                           ("structure-map", 20, "Protein-derived Q3 fractions and structure-tendency field")]:
        item = commands.add_parser(name, help=explanation)
        item.add_argument("--input", required=True, type=Path)
        item.add_argument("--sequence-column", default="Sequence")
        item.add_argument("--pca", required=True, type=Path)
        item.add_argument("--output-dir", required=True, type=Path)
        item.add_argument("--window", type=positive_int, default=50)
        item.add_argument("--step", type=positive_int, default=15)
        item.add_argument("--grid-size", type=float, default=2)
        item.add_argument("--min-points", type=positive_int, default=default_min)
        item.add_argument("--value-limits", nargs=2, type=float, help="Color limits for lag variance or structure tendency")
        if name == "lag-map":
            item.add_argument("--max-lag", type=positive_int, default=30)
            item.add_argument("--unknown-policy", choices=["error", "mma"], default="error")
            item.add_argument("--entropy-limits", nargs=2, type=float, default=[3.388, 3.395])
        item.add_argument("--cmap", default="RdBu_r")
        plot_arguments(item); common_output(item)

    score = commands.add_parser("score-rhp", help="Sample the protein-derived structure field with supplied RHP embeddings")
    score.add_argument("--input", nargs="+", required=True, type=Path)
    score.add_argument("--pca", required=True, type=Path)
    score.add_argument("--map", required=True, type=Path)
    score.add_argument("--output-dir", required=True, type=Path)
    score.add_argument("--pc2-limit", type=float, help="Optional |PC2| restriction; omitted in the uploaded algorithm")
    score.add_argument("--legacy-grid-size", type=float, default=2, help="For an old map with one row/column and no sidecar")
    score.add_argument("--font", default="Arial")
    common_output(score)

    block = commands.add_parser("pseudo-block", help="Generate the uploaded pseudo-block control (independent of Ivan's simulator)")
    block.add_argument("--output", type=Path, required=True)
    block.add_argument("--count", type=positive_int, default=100000)
    block.add_argument("--length", type=positive_int, default=150, help="Nominal length before independent count fluctuations")
    block.add_argument("--composition", nargs=4, type=float, default=[50, 25, 20, 5], metavar=("MMA", "OEGMA", "EHMA", "SPMA"))
    block.add_argument("--seed", type=int, default=777)
    block.add_argument("--global-fluct", type=float, default=.30)
    block.add_argument("--hydrophobic-length", type=positive_int, default=9)
    block.add_argument("--hydrophilic-length", type=positive_int, default=9)
    block.add_argument("--hydrophobic-fluct", type=float, default=.1)
    block.add_argument("--hydrophilic-fluct", type=float, default=.1)
    common_output(block)
    return parser


def convert_windows(args):
    records = []
    prepare_outputs([args.output, meta_path(args.output)], args.overwrite, args.input)
    for path in args.input:
        frame = read_table(path)
        if args.sequence_column not in frame:
            raise ValueError(f"{path}: missing {args.sequence_column}")
        for row_index, row in frame.iterrows():
            encoded = encode_sequence(row[args.sequence_column], unknown_policy=args.unknown_policy)
            for start in sliding_positions(len(encoded), args.window, args.step):
                records.append({"sequence": encoded[start:start + args.window], "source_file": path.name,
                                "source_row": row_index, "Entry": row.get("Entry", str(row_index)),
                                "start": start, "end": start + args.window})
    result = pd.DataFrame(records, columns=["sequence", "source_file", "source_row", "Entry", "start", "end"])
    result.to_csv(args.output, index=False)
    save_json(meta_path(args.output), {"inputs": input_records(args.input), "window": args.window,
                                      "step": args.step, "unknown_policy": args.unknown_policy, "complete_windows": len(result)})
    print(f"Saved {len(result)} complete {args.window}-mer windows to {args.output}")


def generate_blocks(args):
    from ch3lib.blocks import generate_chain
    if any(not 0 <= x < 1 for x in [args.global_fluct, args.hydrophobic_fluct, args.hydrophilic_fluct]):
        raise ValueError("Fluctuations must lie in [0, 1)")
    if any(not math.isfinite(x) or x < 0 for x in args.composition) or sum(args.composition) <= 0:
        raise ValueError("Composition must have nonnegative entries and a positive sum")
    ratios = [x / sum(args.composition) for x in args.composition]
    metrics = Path(str(args.output) + ".chains.csv")
    prepare_outputs([args.output, metrics, meta_path(args.output)], args.overwrite)
    total_counts = Counter()
    length_range = [float("inf"), 0]
    with args.output.open("w", newline="", encoding="utf-8") as sequences, metrics.open("w", newline="", encoding="utf-8") as detail:
        writer, stats = csv.writer(sequences), csv.writer(detail)
        writer.writerow(["Sequence"])
        stats.writerow(["chain_index", "seed", "length", "hydrophobic_blocks", "hydrophilic_blocks", "MMA", "OEGMA", "EHMA", "SPMA"])
        for index in range(args.count):
            seed = args.seed + index
            sequence, info = generate_chain(args.length, ratios, args.global_fluct, args.hydrophobic_length,
                                           args.hydrophobic_fluct, args.hydrophilic_length, args.hydrophilic_fluct, seed)
            writer.writerow([sequence])
            stats.writerow([index, seed, info["length"], info["hydrophobic_blocks"], info["hydrophilic_blocks"]] +
                           [info["counts"][i] for i in range(1, 5)])
            total_counts.update(sequence)
            length_range[0] = min(length_range[0], len(sequence)); length_range[1] = max(length_range[1], len(sequence))
    save_json(meta_path(args.output), {"generator": "uploaded pseudo-block control", "count": args.count, "seed": args.seed,
                                      "nominal_length": args.length, "observed_length_range": length_range,
                                      "composition_order": ["MMA", "OEGMA", "EHMA", "SPMA"], "ratios": ratios,
                                      "global_fluctuation": args.global_fluct, "hydrophobic_length": args.hydrophobic_length,
                                      "hydrophilic_length": args.hydrophilic_length, "hydrophobic_fluctuation": args.hydrophobic_fluct,
                                      "hydrophilic_fluctuation": args.hydrophilic_fluct, "total_monomer_counts": dict(total_counts)})
    print(f"Saved {args.count} pseudo-block chains to {args.output}; lengths {length_range}")


def import_rhpapp(args):
    if len(set(args.source_codes)) != 4:
        raise ValueError("The four source monomer IDs must be distinct")
    mapping = dict(zip(args.source_codes, "1234"))
    prepare_outputs([args.output, meta_path(args.output)], args.overwrite, args.input)
    count = 0
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=args.output.parent, suffix=".partial", delete=False) as output:
            temporary = Path(output.name)
            writer = csv.writer(output)
            writer.writerow(["Sequence", "source_file", "source_row"])
            for path in args.input:
                with path.open(newline="", encoding="utf-8-sig") as source:
                    for row_index, row in enumerate(csv.reader(source)):
                        row = [code.strip() for code in row]
                        while row and not row[-1]: row.pop()
                        if not row: continue
                        unknown = set(row) - set(mapping)
                        if unknown:
                            raise ValueError(f"{path}, row {row_index + 1}: IDs {sorted(unknown)} do not match --source-codes; input must be headerless")
                        writer.writerow(["".join(mapping[code] for code in row), path.name, row_index])
                        count += 1
        temporary.replace(args.output)
    finally:
        if temporary is not None and temporary.exists(): temporary.unlink()
    save_json(meta_path(args.output), {"source": "RHPapp headerless monomer-array export", "inputs": input_records(args.input),
                                      "source_id_to_target_code": mapping, "target_code_order": ["MMA", "OEGMA", "EHMA", "SPMA"],
                                      "chain_count": count})
    print(f"Imported {count} RHPapp chains into {args.output}")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "windows":
            convert_windows(args)
        elif args.command == "import-rhpapp":
            import_rhpapp(args)
        elif args.command == "pseudo-block":
            generate_blocks(args)
        elif args.command == "embed":
            from ch3lib.inference import embed_files
            embed_files(args.input, args.output_dir, args.ae_repo, args.checkpoint, args.kind, args.sequence_column,
                        args.step, args.batch_size, args.device, args.unknown_policy, args.overwrite, args.model_config)
        elif args.command == "predict-q3":
            from ch3lib.inference import predict_q3_files
            predict_q3_files(args.input, args.output_dir, args.esm_checkpoint, args.porter_source, args.porter_checkpoint,
                             args.device, args.chunk_size, args.overwrite, args.on_error, args.sequence_column)
        elif args.command == "fit-pca":
            from ch3lib.analysis import fit_pca
            if args.groups:
                from ch3lib.plotting import load_groups
                groups = [g for g in load_groups(args.groups) if g.get("fit_reference", False)]
                if any(g["kind"] != "protein" for g in groups):
                    raise ValueError("The protein-derived group reference must contain only kind=protein inputs")
                paths = [g["file"] for g in groups]
                if not paths: raise ValueError("No fit_reference=true groups; choose reference inputs explicitly")
            else:
                paths = args.input
            fit_pca(paths, args.output, args.components, args.overwrite)
            print(f"Saved fixed PCA reference to {args.output}")
        elif args.command == "plot-kde":
            from ch3lib.plotting import plot_kde
            plot_kde(args.groups, args.pca, args.output_dir, args.threshold, args.font, args.overwrite, args.xlim, args.ylim)
        elif args.command in {"lag-map", "structure-map"}:
            from ch3lib.analysis import make_map
            from ch3lib.plotting import heatmap
            kind = "lag" if args.command == "lag-map" else "structure"
            columns = [("variance_of_mean_lag_profile", "lag_variance", "Lag variance", (0, .15)),
                       ("mean_lag_entropy", "lag_entropy", "Mean lag-profile entropy", (3.388, 3.395))] if kind == "lag" else [
                           ("structure_tendency", "structure_tendency", "Structure tendency score", (-1, 1))]
            if args.window != 50:
                raise ValueError("The supplied AE_MP embedding workflow uses 50-mer map windows")
            if args.value_limits is not None:
                columns[0] = (*columns[0][:3], args.value_limits)
            if kind == "lag":
                columns[1] = (*columns[1][:3], args.entropy_limits)
            figures = [args.output_dir / (stem + suffix) for _, stem, _, _ in columns for suffix in (".png", ".pdf")]
            prepare_outputs(figures, args.overwrite, [args.input, args.pca])
            frame, grid, output = make_map(args.input, args.pca, args.output_dir, kind, args.grid_size, args.min_points,
                                           args.window, args.step, getattr(args, "max_lag", 30), args.overwrite,
                                           getattr(args, "unknown_policy", "error"), args.sequence_column)
            for column, stem, label, limits in columns:
                heatmap(frame, grid, column, args.output_dir / stem, label, limits, args.font, args.overwrite, args.xlim, args.ylim, args.cmap)
            print(f"Saved {output}")
        elif args.command == "score-rhp":
            from ch3lib.analysis import score_rhp
            from ch3lib.plotting import rank_curves
            _, points = score_rhp(args.input, args.pca, args.map, args.output_dir, args.overwrite, args.pc2_limit, args.legacy_grid_size)
            rank_curves(points, args.output_dir, args.font, args.overwrite)
            print(f"Saved RHP score summary and rank curves to {args.output_dir}")
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
