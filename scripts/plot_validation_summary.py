#!/usr/bin/env python3
"""
Generate visualization plots from multi-case validation results.

Usage:
    python scripts/plot_validation_summary.py \\
        --csv results/validation_summary/multi_case_metrics.csv \\
        --output-dir results/validation_summary/figures

Outputs:
    - dice_summary.png (boxplot/barplot of Dice scores)
    - hd95_summary.png (HD95 per case)
    - volume_diff_summary.png (volume difference percentage)
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate validation summary plots")

    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/validation_summary/multi_case_metrics.csv"),
        help="Path to metrics CSV file",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/validation_summary/figures"),
        help="Output directory for figures",
    )

    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"], help="Output format (default: png)")

    parser.add_argument("--dpi", type=int, default=150, help="DPI for output images (default: 150)")

    return parser.parse_args()


def load_data(csv_path: Path) -> Optional[pd.DataFrame]:
    """Load and validate metrics data."""
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path)

        # Filter only successful cases
        if "status" in df.columns:
            df = df[df["status"] == "success"]

        if len(df) == 0:
            print("❌ No successful cases found in CSV")
            return None

        print(f"✅ Loaded {len(df)} successful cases from {csv_path}")
        return df

    except Exception as exc:
        print(f"❌ Error loading CSV: {exc}")
        return None


def plot_dice_scores(df: pd.DataFrame, output_dir: Path, fmt: str, dpi: int) -> None:
    """Generate Dice scores summary plot."""

    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data
    case_ids = df["case_id"].values
    x = np.arange(len(case_ids))
    width = 0.25

    dice_wt = df["dice_wt"].values if "dice_wt" in df.columns else None
    dice_tc = df["dice_tc"].values if "dice_tc" in df.columns else None
    dice_et = df["dice_et"].values if "dice_et" in df.columns else None

    # Plot bars
    if dice_wt is not None:
        ax.bar(x - width, dice_wt, width, label="WT", color="#2E86AB", alpha=0.8)
    if dice_tc is not None:
        ax.bar(x, dice_tc, width, label="TC", color="#A23B72", alpha=0.8)
    if dice_et is not None:
        ax.bar(x + width, dice_et, width, label="ET", color="#F18F01", alpha=0.8)

    # Add mean line
    if dice_wt is not None and len(dice_wt) > 0:
        mean_wt = np.mean(dice_wt)
        ax.axhline(y=mean_wt, color="#2E86AB", linestyle="--", linewidth=2, alpha=0.7)
        ax.text(
            len(case_ids) - 0.5, mean_wt + 0.02, f"μ={mean_wt:.3f}", fontsize=10, color="#2E86AB", fontweight="bold"
        )

    # Formatting
    ax.set_xlabel("Case ID", fontsize=12, fontweight="bold")
    ax.set_ylabel("Dice Score", fontsize=12, fontweight="bold")
    ax.set_title("Dice Scores by Case and Sub-region", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(case_ids, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()

    output_path = output_dir / f"dice_summary.{fmt}"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"✅ Dice summary plot saved: {output_path}")


def plot_hd95(df: pd.DataFrame, output_dir: Path, fmt: str, dpi: int) -> None:
    """Generate HD95 summary plot."""

    if "hd95_wt" not in df.columns:
        print("⚠️ HD95 data not available, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    case_ids = df["case_id"].values
    hd95 = df["hd95_wt"].values

    # Bar plot
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(hd95)))
    bars = ax.bar(case_ids, hd95, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

    # Add mean line
    mean_hd95 = np.mean(hd95)
    ax.axhline(y=mean_hd95, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_hd95:.2f} mm")

    # Color coding for interpretation
    for i, (bar, val) in enumerate(zip(bars, hd95)):
        if val < 10:
            bar.set_facecolor("#2E7D32")  # Green - excellent
        elif val < 20:
            bar.set_facecolor("#F57C00")  # Orange - good
        else:
            bar.set_facecolor("#C62828")  # Red - review needed

    # Formatting
    ax.set_xlabel("Case ID", fontsize=12, fontweight="bold")
    ax.set_ylabel("HD95 (mm)", fontsize=12, fontweight="bold")
    ax.set_title("Hausdorff Distance 95th Percentile by Case", fontsize=14, fontweight="bold")
    ax.set_xticklabels(case_ids, rotation=45, ha="right", fontsize=9)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add legend for colors
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2E7D32", label="Excellent (< 10 mm)"),
        Patch(facecolor="#F57C00", label="Good (10-20 mm)"),
        Patch(facecolor="#C62828", label="Review needed (> 20 mm)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    plt.tight_layout()

    output_path = output_dir / f"hd95_summary.{fmt}"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"✅ HD95 summary plot saved: {output_path}")


def plot_volume_diff(df: pd.DataFrame, output_dir: Path, fmt: str, dpi: int) -> None:
    """Generate volume difference plot."""

    if "volume_diff_percent" not in df.columns:
        print("⚠️ Volume difference data not available, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    case_ids = df["case_id"].values
    vol_diff = df["volume_diff_percent"].values

    # Bar plot
    colors = ["#2E7D32" if abs(v) < 10 else "#F57C00" if abs(v) < 20 else "#C62828" for v in vol_diff]
    ax.bar(case_ids, vol_diff, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

    # Zero line
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1)

    # ±10% and ±20% reference lines
    ax.axhline(y=10, color="orange", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(y=-10, color="orange", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(y=20, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(y=-20, color="red", linestyle="--", linewidth=1, alpha=0.5)

    # Formatting
    ax.set_xlabel("Case ID", fontsize=12, fontweight="bold")
    ax.set_ylabel("Volume Difference (%)", fontsize=12, fontweight="bold")
    ax.set_title("Predicted vs Ground Truth Volume Difference", fontsize=14, fontweight="bold")
    ax.set_xticklabels(case_ids, rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2E7D32", label="< 10% (Excellent)"),
        Patch(facecolor="#F57C00", label="10-20% (Acceptable)"),
        Patch(facecolor="#C62828", label="> 20% (Review)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    plt.tight_layout()

    output_path = output_dir / f"volume_diff_summary.{fmt}"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"✅ Volume difference plot saved: {output_path}")


def plot_overall_summary(df: pd.DataFrame, output_dir: Path, fmt: str, dpi: int) -> None:
    """Generate overall summary dashboard."""

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle("PY-BRAIN Multi-Case Validation Summary", fontsize=16, fontweight="bold", y=0.98)

    # 1. Dice scores comparison
    ax1 = fig.add_subplot(gs[0, :])
    case_ids = df["case_id"].values
    x = np.arange(len(case_ids))

    if "dice_wt" in df.columns:
        ax1.plot(x, df["dice_wt"].values, "o-", label="WT", linewidth=2, markersize=8, color="#2E86AB")
    if "dice_tc" in df.columns:
        ax1.plot(x, df["dice_tc"].values, "s-", label="TC", linewidth=2, markersize=8, color="#A23B72")
    if "dice_et" in df.columns:
        ax1.plot(x, df["dice_et"].values, "^-", label="ET", linewidth=2, markersize=8, color="#F18F01")

    ax1.set_ylabel("Dice Score", fontweight="bold")
    ax1.set_title("Dice Scores Across Cases", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(case_ids, rotation=45, ha="right")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="lower left")
    ax1.grid(alpha=0.3)
    ax1.axhline(y=0.9, color="green", linestyle="--", alpha=0.5, label="Target (0.9)")

    # 2. Boxplot of Dice scores
    ax2 = fig.add_subplot(gs[1, 0])
    dice_data = []
    dice_labels = []
    if "dice_wt" in df.columns:
        dice_data.append(df["dice_wt"].values)
        dice_labels.append("WT")
    if "dice_tc" in df.columns:
        dice_data.append(df["dice_tc"].values)
        dice_labels.append("TC")
    if "dice_et" in df.columns:
        dice_data.append(df["dice_et"].values)
        dice_labels.append("ET")

    if dice_data:
        bp = ax2.boxplot(dice_data, labels=dice_labels, patch_artist=True)
        colors = ["#2E86AB", "#A23B72", "#F18F01"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.set_ylabel("Dice Score", fontweight="bold")
        ax2.set_title("Distribution of Dice Scores", fontweight="bold")
        ax2.set_ylim(0, 1.05)
        ax2.grid(axis="y", alpha=0.3)

    # 3. HD95 vs Dice scatter
    ax3 = fig.add_subplot(gs[1, 1])
    if "hd95_wt" in df.columns and "dice_wt" in df.columns:
        ax3.scatter(df["dice_wt"], df["hd95_wt"], s=100, alpha=0.6, c=range(len(df)), cmap="viridis")
        for i, case_id in enumerate(case_ids):
            ax3.annotate(case_id.split("_")[-1], (df["dice_wt"].iloc[i], df["hd95_wt"].iloc[i]), fontsize=8, alpha=0.7)
        ax3.set_xlabel("Dice WT", fontweight="bold")
        ax3.set_ylabel("HD95 (mm)", fontweight="bold")
        ax3.set_title("Dice vs HD95 Trade-off", fontweight="bold")
        ax3.grid(alpha=0.3)

    # 4. Runtime analysis
    ax4 = fig.add_subplot(gs[2, 0])
    if "runtime_seconds" in df.columns:
        runtimes = df["runtime_seconds"].values / 60  # Convert to minutes
        ax4.bar(case_ids, runtimes, color="#6A4C93", alpha=0.8)
        ax4.set_ylabel("Runtime (minutes)", fontweight="bold")
        ax4.set_title("Pipeline Runtime per Case", fontweight="bold")
        ax4.set_xticklabels(case_ids, rotation=45, ha="right")
        ax4.grid(axis="y", alpha=0.3)

    # 5. Summary statistics table
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")

    # Create summary text
    summary_text = "Summary Statistics\n" + "=" * 30 + "\n\n"

    if "dice_wt" in df.columns:
        summary_text += "Dice WT:\n"
        summary_text += f"  Mean:  {df['dice_wt'].mean():.4f}\n"
        summary_text += f"  Std:   {df['dice_wt'].std():.4f}\n"
        summary_text += f"  Range: {df['dice_wt'].min():.4f} - {df['dice_wt'].max():.4f}\n\n"

    if "hd95_wt" in df.columns:
        summary_text += "HD95:\n"
        summary_text += f"  Mean:  {df['hd95_wt'].mean():.2f} mm\n"
        summary_text += f"  Std:   {df['hd95_wt'].std():.2f} mm\n\n"

    if "volume_diff_percent" in df.columns:
        summary_text += "Volume Diff:\n"
        summary_text += f"  Mean:  {df['volume_diff_percent'].mean():.1f}%\n"
        summary_text += f"  Std:   {df['volume_diff_percent'].std():.1f}%\n"

    ax5.text(
        0.1,
        0.9,
        summary_text,
        transform=ax5.transAxes,
        fontsize=11,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()

    output_path = output_dir / f"overall_summary.{fmt}"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"✅ Overall summary dashboard saved: {output_path}")


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data(args.csv)
    if df is None:
        return 1

    print(f"\n{'=' * 60}")
    print("Generating validation summary plots")
    print(f"{'=' * 60}")

    # Generate plots
    plot_dice_scores(df, args.output_dir, args.format, args.dpi)
    plot_hd95(df, args.output_dir, args.format, args.dpi)
    plot_volume_diff(df, args.output_dir, args.format, args.dpi)
    plot_overall_summary(df, args.output_dir, args.format, args.dpi)

    print(f"\n{'=' * 60}")
    print(f"✅ All plots saved to: {args.output_dir}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
