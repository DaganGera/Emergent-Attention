"""
Generate LaTeX tables from experiment results.

Reads JSON result files from results/ and generates a formatted LaTeX table
with accuracy ± std, FLOPs, params, and throughput. Best results are bolded.

Usage:
    python scripts/generate_latex_tables.py --results-dir results/ --output tables.tex

    # With W&B API (reads multiple seeds for mean ± std):
    python scripts/generate_latex_tables.py --from-wandb --project emergent-attention
"""

import os
import sys
import json
import argparse
import glob
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def bold(val: str) -> str:
    return r"\textbf{" + val + r"}"


def format_acc(mean: float, std: float | None = None) -> str:
    if std is not None and std > 0:
        return f"{mean:.1f}$\\pm${std:.1f}"
    return f"{mean:.1f}"


def generate_table_from_results(results_dir: str, dataset: str = "cifar100") -> str:
    """
    Build LaTeX table from JSON result files in results_dir.

    Expects files named like: nca_vit_tiny_cifar100_seed42.json
    Groups multiple seeds and computes mean ± std.
    """
    # Load all result files
    files = glob.glob(os.path.join(results_dir, f"*_{dataset}*.json"))
    if not files:
        raise FileNotFoundError(f"No result files found for dataset={dataset} in {results_dir}")

    # Group by model name
    by_model: dict[str, list[dict]] = defaultdict(list)
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        by_model[data["model"]].append(data)

    # Display order
    display_order = ["vit_tiny", "deit_tiny", "swin_tiny", "nca_vit_hybrid", "nca_vit_tiny"]
    display_names = {
        "vit_tiny": "ViT-Tiny",
        "deit_tiny": "DeiT-Tiny",
        "swin_tiny": "Swin-Tiny",
        "nca_vit_hybrid": "NCA-ViT-Hybrid (ours)",
        "nca_vit_tiny": r"\textbf{NCA-ViT-Tiny (ours)}",
    }

    rows = []
    col_values: dict[str, list] = defaultdict(list)  # for bolding

    for model_key in display_order:
        if model_key not in by_model:
            continue
        runs = by_model[model_key]
        top1s = [r["top1"] for r in runs if "top1" in r]
        top5s = [r["top5"] for r in runs if "top5" in r]
        gflops = runs[0].get("gflops", None)
        params_M = runs[0].get("params_M", None)
        ips = runs[0].get("images_per_sec", None)

        mean_top1 = np.mean(top1s) if top1s else None
        std_top1 = np.std(top1s) if len(top1s) > 1 else None
        mean_top5 = np.mean(top5s) if top5s else None

        rows.append({
            "model": model_key,
            "display_name": display_names.get(model_key, model_key),
            "top1": mean_top1,
            "top1_std": std_top1,
            "top5": mean_top5,
            "gflops": gflops,
            "params_M": params_M,
            "images_per_sec": ips,
        })
        if mean_top1 is not None:
            col_values["top1"].append(mean_top1)
        if gflops is not None:
            col_values["gflops"].append(gflops)

    if not rows:
        raise ValueError("No data to generate table")

    best_top1 = max(col_values["top1"]) if col_values["top1"] else None
    best_gflops = min(col_values["gflops"]) if col_values["gflops"] else None

    # Build LaTeX
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Comparison on " + dataset.upper() + r"-100. Results averaged over 3 seeds ($\pm$std).}",
        r"  \label{tab:main-results}",
        r"  \begin{tabular}{lcccc}",
        r"    \toprule",
        r"    Model & Top-1 (\%) & GFLOPs & Params (M) & Img/s \\",
        r"    \midrule",
    ]

    for row in rows:
        name = row["display_name"]

        # Top-1
        if row["top1"] is not None:
            acc_str = format_acc(row["top1"], row["top1_std"])
            if best_top1 is not None and abs(row["top1"] - best_top1) < 0.05:
                acc_str = bold(acc_str)
        else:
            acc_str = "--"

        # GFLOPs
        if row["gflops"] is not None:
            flop_str = f"{row['gflops']:.2f}"
            if best_gflops is not None and abs(row["gflops"] - best_gflops) < 0.01:
                flop_str = bold(flop_str)
        else:
            flop_str = "--"

        params_str = f"{row['params_M']:.1f}" if row["params_M"] is not None else "--"
        ips_str = f"{row['images_per_sec']:.0f}" if row["images_per_sec"] is not None else "--"

        lines.append(f"    {name} & {acc_str} & {flop_str} & {params_str} & {ips_str} \\\\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--dataset", default="cifar100", choices=["cifar100", "imagenet100"])
    parser.add_argument("--output", default=None, help="Output .tex file (prints to stdout if not set)")
    args = parser.parse_args()

    table = generate_table_from_results(args.results_dir, args.dataset)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(table)
        print(f"Table written to: {args.output}")
    else:
        print(table)


if __name__ == "__main__":
    main()
