"""
Unified pass@k plotting script supporting multiple use cases:
1. Single model with original distribution
2. Single model with tilted distributions (beta1/beta2 variations)
3. Multiple models/checkpoints with original distribution
4. Multiple models/checkpoints with tilted distributions
"""

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_data(result_files: List[str], labels=None) -> pd.DataFrame:
    dfs = []
    for result_file in result_files:
        filepath = Path(result_file)
        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping")
            continue

        df = pd.read_csv(filepath)

        # Add source identifier
        df["source"] = result_file
        if labels:
            df["label"] = labels[result_file]
        dfs.append(df)

    if not dfs:
        raise ValueError("No valid data files found")

    return pd.concat(dfs, ignore_index=True)


def configure_x_axis(df: pd.DataFrame, max_ticks: int = 10):
    """Configure x-axis to show evenly spaced k values.

    Args:
        df: DataFrame containing k values
        max_ticks: Maximum number of ticks to show on x-axis
    """
    k_values = sorted(df["k"].unique())
    if len(k_values) <= max_ticks:
        tick_positions = k_values
    else:
        step = len(k_values) // (max_ticks - 1)
        tick_positions = k_values[::step]
        if k_values[-1] not in tick_positions:
            tick_positions.append(k_values[-1])
    plt.xticks(tick_positions, fontsize=10)


def make_plot(df: pd.DataFrame, metric: str, output_path: str, title: Optional[str] = None, legend_title: Optional[str] = None, figsize: tuple = (10, 6)):
    if metric not in df.columns:
        print(f"Metric {metric} not found in data, skipping")
        return

    sns.set(style="whitegrid")
    plt.figure(figsize=figsize)

    # Group by source for multi-model plots
    for source, group in df.groupby("source"):
        if "label" in group.columns:
            label = group.iloc[0]["label"]
        else:
            label = source
        sns.lineplot(x="k", y=metric, data=group, label=label, marker="o")

    if title:
        plt.title(title)

    plt.xlabel("K")
    ylabel = "Pass@K" if metric == "pass_at_k_mean" else "Avg@K"
    plt.ylabel(ylabel)

    configure_x_axis(df)

    if legend_title:
        plt.legend(title=legend_title)
    else:
        plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.clf()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Create pass@k plots from evaluation results", formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--result_files", type=str, required=True, help="comma-separated list of result CSV files to include in the plot")

    parser.add_argument("--output_dir", required=True, help="Directory to save output plots")

    parser.add_argument("--labels", type=str, default=None, help="Comma-separated list of labels for each result file")

    parser.add_argument("--title", help="Plot title (optional)")

    parser.add_argument("--legend_title", help="Legend title (optional)")

    parser.add_argument("--figsize", nargs=2, type=float, default=[10, 6], help="Figure size as width height. Default: 10 6")

    parser.add_argument("--prefix", default="pass_at_k_plot", help="Prefix for output filenames. Default: pass_at_k_plot")

    args = parser.parse_args()
    args.result_files = args.result_files.split(",")  # Split comma-separated list into Python list
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    if args.labels:
        labels = args.labels.split(",")
        assert len(labels) == len(args.result_files), "Number of labels must match number of result directories"
        labels = {result_file: label for result_file, label in zip(args.result_files, labels)}
    else:
        labels = None
    df = load_data(args.result_files, labels=labels)

    print(f"Loaded {len(df)} rows from {len(args.result_files)} directories")

    # Create pass@k plot
    pass_at_k_output = output_dir / "pass_at_k.png"
    make_plot(df=df, metric="pass_at_k_mean", output_path=str(pass_at_k_output), title=args.title, legend_title=args.legend_title, figsize=tuple(args.figsize))

    # Create avg@k plot if data is available
    if "avg_at_k_mean" in df.columns:
        avg_at_k_output = output_dir / "avg_at_k.png"
        avg_title = args.title.replace("Pass@K", "Avg@K") if args.title else None
        make_plot(df=df, metric="avg_at_k_mean", output_path=str(avg_at_k_output), title=avg_title, legend_title=args.legend_title, figsize=tuple(args.figsize))


if __name__ == "__main__":
    main()
