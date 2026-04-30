#!/usr/bin/env python3
"""
Multi-Case Validation Workflow for PY-BRAIN

Executes the pipeline on multiple BraTS cases and aggregates results
for scientific documentation and reproducible validation studies.

Usage:
    python scripts/run_multi_case_validation.py \
        --brats-root data/datasets/BraTS2021/raw/BraTS2021_Training_Data \
        --cases BraTS2021_00000 BraTS2021_00002 BraTS2021_00003 \
        --output-dir results/validation_runs \
        --summary-dir results/validation_summary \
        --device mps

Outputs:
    - Per-case sessions in results/validation_runs/<case_id>/
    - Aggregated metrics CSV, JSON, and Markdown reports
    - Summary visualizations
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("multi_case_validation")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-case validation for PY-BRAIN pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run specific cases
    python scripts/run_multi_case_validation.py \\
        --brats-root data/datasets/BraTS2021/raw/BraTS2021_Training_Data \\
        --cases BraTS2021_00000 BraTS2021_00002 BraTS2021_00003

    # Auto-discover first N cases
    python scripts/run_multi_case_validation.py \\
        --brats-root data/datasets/BraTS2021/raw/BraTS2021_Training_Data \\
        --auto-discover 10

    # Use specific device
    python scripts/run_multi_case_validation.py \\
        --brats-root /path/to/BraTS \\
        --cases BraTS2021_00000 \\
        --device cpu \\
        --skip-existing
        """,
    )

    parser.add_argument("--brats-root", type=Path, required=True, help="Root directory containing BraTS case folders")

    parser.add_argument("--cases", nargs="+", help="List of case IDs to process (e.g., BraTS2021_00000)")

    parser.add_argument("--auto-discover", type=int, metavar="N", help="Auto-discover and run first N valid cases")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/validation_runs"),
        help="Directory for per-case outputs (default: results/validation_runs)",
    )

    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("results/validation_summary"),
        help="Directory for aggregated summary outputs (default: results/validation_summary)",
    )

    parser.add_argument(
        "--device", default="mps", choices=["cpu", "cuda", "mps"], help="Device for inference (default: mps)"
    )

    parser.add_argument("--skip-existing", action="store_true", help="Skip cases that already have completed outputs")

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue with remaining cases if one fails (default: True)",
    )

    parser.add_argument("--timeout", type=int, default=600, help="Timeout per case in seconds (default: 600)")

    return parser.parse_args()


def discover_cases(brats_root: Path, n_cases: int) -> List[str]:
    """Auto-discover valid BraTS cases."""
    logger.info(f"Auto-discovering cases in {brats_root}")

    cases = []
    for case_dir in sorted(brats_root.iterdir()):
        if not case_dir.is_dir():
            continue

        case_id = case_dir.name
        seg_file = case_dir / f"{case_id}_seg.nii.gz"

        # Check for required files
        required_files = [
            seg_file,
            case_dir / f"{case_id}_flair.nii.gz",
            case_dir / f"{case_id}_t1.nii.gz",
            case_dir / f"{case_id}_t1ce.nii.gz",
            case_dir / f"{case_id}_t2.nii.gz",
        ]

        if all(f.exists() for f in required_files):
            cases.append(case_id)

        if len(cases) >= n_cases:
            break

    logger.info(f"Discovered {len(cases)} valid cases")
    return cases


def prepare_case(case_id: str, brats_root: Path, output_dir: Path, device: str) -> Tuple[bool, Path, Optional[str]]:
    """
    Prepare a BraTS case for pipeline execution.

    Returns:
        (success: bool, session_json_path: Path, error_message: Optional[str])
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Preparing case: {case_id}")
    logger.info(f"{'=' * 60}")

    case_output_dir = output_dir / case_id
    case_output_dir.mkdir(parents=True, exist_ok=True)

    session_json = case_output_dir / "session.json"

    # Run prepare_brats_case.py (always run to ensure files are present)
    project_root = Path(__file__).parent.parent
    prepare_script = project_root / "prepare_brats_case.py"

    cmd = [
        sys.executable,
        str(prepare_script),
        "--brats-root",
        str(brats_root),
        "--case",
        case_id,
        "--output",
        str(case_output_dir),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            error_msg = f"prepare_brats_case failed: {result.stderr}"
            logger.error(error_msg)
            return False, case_output_dir, error_msg

        # Create or update session.json with required fields
        # Remove old file to ensure clean state
        if session_json.exists():
            session_json.unlink()

        session_data = {
            "case_id": case_id,
            "session_id": case_id,
            "monai_dir": str(case_output_dir),
            "nifti_dir": str(case_output_dir),
            "extra_dir": str(case_output_dir / "extra_sequences"),
            "output_dir": str(case_output_dir),
            "results_dir": str(case_output_dir.parent),
            "project_root": str(project_root),
            "bundle_dir": str(project_root / "models" / "brats_bundle"),
            "ground_truth": str(case_output_dir / "ground_truth.nii.gz"),
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        }
        # Create extra_dir if needed
        Path(session_data["extra_dir"]).mkdir(parents=True, exist_ok=True)
        with open(session_json, "w") as f:
            json.dump(session_data, f, indent=2)
        logger.info(f"Created session.json with monai_dir: {case_output_dir}")

        logger.info("✅ Case prepared successfully")
        return True, session_json, None

    except subprocess.TimeoutExpired:
        error_msg = "Preparation timeout (60s)"
        logger.error(error_msg)
        return False, case_output_dir, error_msg
    except Exception as exc:
        error_msg = f"Preparation error: {exc}"
        logger.error(error_msg)
        return False, case_output_dir, error_msg


def run_pipeline(
    session_json: Path, case_output_dir: Path, device: str, timeout: int
) -> Tuple[bool, Optional[str], float]:
    """
    Run the main pipeline for a prepared case.

    Returns:
        (success: bool, error_message: Optional[str], runtime_seconds: float)
    """
    logger.info(f"Running pipeline for: {session_json.parent.name}")

    log_file = case_output_dir / "run.log"

    # Check for existing successful run
    validation_metrics = case_output_dir / "validation_metrics.json"
    if validation_metrics.exists():
        logger.info("Validation metrics already exist, skipping")
        return True, None, 0.0

    # Set environment
    env = {
        **dict(subprocess.os.environ),
        "PYBRAIN_SESSION": str(session_json),
    }

    cmd = [
        sys.executable,
        "scripts/3_brain_tumor_analysis.py",
    ]

    start_time = time.time()

    try:
        with open(log_file, "w") as log_fh:
            result = subprocess.run(
                cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env, timeout=timeout, cwd=Path(__file__).parent.parent
            )

        runtime = time.time() - start_time

        if result.returncode != 0:
            error_msg = f"Pipeline failed with code {result.returncode}"
            logger.error(error_msg)
            return False, error_msg, runtime

        # Verify outputs
        if not validation_metrics.exists():
            error_msg = "Pipeline completed but validation_metrics.json not found"
            logger.error(error_msg)
            return False, error_msg, runtime

        logger.info(f"✅ Pipeline completed in {runtime:.1f}s")
        return True, None, runtime

    except subprocess.TimeoutExpired:
        runtime = time.time() - start_time
        error_msg = f"Pipeline timeout ({timeout}s)"
        logger.error(error_msg)
        return False, error_msg, runtime
    except Exception as exc:
        runtime = time.time() - start_time
        error_msg = f"Pipeline error: {exc}"
        logger.error(error_msg)
        return False, error_msg, runtime


def extract_metrics(case_output_dir: Path) -> Optional[Dict[str, Any]]:
    """Extract metrics from validation_metrics.json."""
    metrics_file = case_output_dir / "validation_metrics.json"

    if not metrics_file.exists():
        return None

    try:
        with open(metrics_file) as f:
            data = json.load(f)

        # Extract key metrics
        metrics = {
            "case_id": case_output_dir.name,
            "dice_wt": data.get("dice_wt"),
            "dice_tc": data.get("dice_tc"),
            "dice_et": data.get("dice_et"),
            "hd95_wt": data.get("hd95_wt"),
            "asd": data.get("asd"),
            "volume_pred_cc": data.get("volume_pred_cc"),
            "volume_gt_cc": data.get("volume_gt_cc"),
            "volume_diff_percent": data.get("volume_diff_percent"),
        }

        return metrics

    except Exception as exc:
        logger.warning(f"Failed to extract metrics from {metrics_file}: {exc}")
        return None


def aggregate_results(case_results: List[Dict[str, Any]], summary_dir: Path) -> None:
    """Aggregate results into CSV, JSON, and Markdown reports."""

    logger.info(f"\n{'=' * 60}")
    logger.info("Aggregating results")
    logger.info(f"{'=' * 60}")

    summary_dir.mkdir(parents=True, exist_ok=True)

    # Create DataFrame
    df = pd.DataFrame(case_results)

    # Calculate summary statistics
    numeric_cols = ["dice_wt", "dice_tc", "dice_et", "hd95_wt", "asd", "volume_diff_percent"]
    stats = {}

    for col in numeric_cols:
        if col in df.columns:
            values = df[col].dropna()
            if len(values) > 0:
                stats[col] = {
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }

    # Save CSV
    csv_path = summary_dir / "multi_case_metrics.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"✅ CSV saved: {csv_path}")

    # Save JSON summary
    json_path = summary_dir / "multi_case_summary.json"
    summary_data = {
        "n_cases": len(case_results),
        "n_successful": len([r for r in case_results if r.get("status") == "success"]),
        "n_failed": len([r for r in case_results if r.get("status") == "failed"]),
        "statistics": stats,
        "cases": case_results,
    }

    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"✅ JSON saved: {json_path}")

    # Generate Markdown report
    generate_markdown_report(df, stats, summary_dir)


def generate_markdown_report(df: pd.DataFrame, stats: Dict[str, Dict[str, float]], summary_dir: Path) -> None:
    """Generate professional Markdown report."""

    report_path = summary_dir / "MULTI_CASE_VALIDATION_REPORT.md"

    n_cases = len(df)
    n_success = len(df[df["status"] == "success"]) if "status" in df.columns else n_cases
    n_failed = n_cases - n_success

    lines = [
        "# Multi-Case Validation Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "**Project:** PY-BRAIN Brain Tumor Segmentation",
        "",
        "## Executive Summary",
        "",
        f"- **Total Cases:** {n_cases}",
        f"- **Successful:** {n_success}",
        f"- **Failed:** {n_failed}",
        "",
        "## Summary Statistics",
        "",
        "| Metric | Mean | Median | Std | Min | Max |",
        "|--------|------|--------|-----|-----|-----|",
    ]

    for metric, values in stats.items():
        lines.append(
            f"| {metric} | "
            f"{values['mean']:.4f} | "
            f"{values['median']:.4f} | "
            f"{values['std']:.4f} | "
            f"{values['min']:.4f} | "
            f"{values['max']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Per-Case Results",
            "",
        ]
    )

    # Add per-case table
    if not df.empty:
        lines.append(df.to_markdown(index=False))

    lines.extend(
        [
            "",
            "## Failed Cases",
            "",
        ]
    )

    if "status" in df.columns and "error" in df.columns:
        failed = df[df["status"] == "failed"]
        if len(failed) > 0:
            for _, row in failed.iterrows():
                lines.append(f"- **{row['case_id']}:** {row.get('error', 'Unknown error')}")
        else:
            lines.append("No failures recorded.")
    else:
        lines.append("No failure data available.")

    lines.extend(
        [
            "",
            "## Technical Conclusions",
            "",
            "### Stability Assessment",
            "",
        ]
    )

    # Stability assessment
    if "dice_wt" in stats:
        dice_std = stats["dice_wt"]["std"]
        if dice_std < 0.02:
            stability = "✅ **Excellent** — Very consistent performance across cases"
        elif dice_std < 0.05:
            stability = "✅ **Good** — Acceptable variability, pipeline is stable"
        elif dice_std < 0.10:
            stability = "⚠️ **Moderate** — Some variability, review challenging cases"
        else:
            stability = "❌ **Poor** — High variability, needs investigation"

        lines.append(f"- **Dice WT Stability:** {stability} (std={dice_std:.4f})")

    lines.extend(
        [
            "",
            "### Recommendations",
            "",
            "- Pipeline is ready for production use."
            if n_failed == 0
            else "- Review failed cases before production deployment.",
            "- Consider expanding validation to more challenging cases.",
            "",
        ]
    )

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"✅ Markdown report saved: {report_path}")


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Validate inputs
    if not args.brats_root.exists():
        logger.error(f"BraTS root directory not found: {args.brats_root}")
        return 1

    # Determine cases to process
    if args.auto_discover:
        cases = discover_cases(args.brats_root, args.auto_discover)
    elif args.cases:
        cases = args.cases
    else:
        logger.error("Either --cases or --auto-discover must be specified")
        return 1

    if not cases:
        logger.error("No cases to process")
        return 1

    logger.info("Multi-case validation started")
    logger.info(f"Cases: {cases}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Summary directory: {args.summary_dir}")

    # Process each case
    case_results = []

    for idx, case_id in enumerate(cases, 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing case {idx}/{len(cases)}: {case_id}")
        logger.info(f"{'=' * 60}")

        case_result = {
            "case_id": case_id,
            "status": "pending",
            "error": None,
            "runtime_seconds": None,
        }

        # Check if already processed (skip-existing)
        validation_metrics_file = args.output_dir / case_id / "validation_metrics.json"
        if args.skip_existing and validation_metrics_file.exists():
            logger.info(f"Skipping {case_id} (validation_metrics.json exists)")
            case_result["status"] = "skipped"
            metrics = extract_metrics(args.output_dir / case_id)
            if metrics:
                case_result.update(metrics)
            case_results.append(case_result)
            continue

        # Prepare case
        success, session_json, error = prepare_case(case_id, args.brats_root, args.output_dir, args.device)

        if not success:
            case_result["status"] = "failed"
            case_result["error"] = error or "Preparation failed"
            case_results.append(case_result)

            if not args.continue_on_error:
                logger.error("Stopping due to failure (continue-on-error=False)")
                break
            continue

        # Run pipeline
        success, error, runtime = run_pipeline(session_json, args.output_dir / case_id, args.device, args.timeout)

        case_result["runtime_seconds"] = runtime

        if success:
            case_result["status"] = "success"

            # Extract metrics
            metrics = extract_metrics(args.output_dir / case_id)
            if metrics:
                case_result.update(metrics)
        else:
            case_result["status"] = "failed"
            case_result["error"] = error or "Pipeline failed"

            if not args.continue_on_error:
                logger.error("Stopping due to failure (continue-on-error=False)")
                break

        case_results.append(case_result)

    # Aggregate results
    if case_results:
        aggregate_results(case_results, args.summary_dir)

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("VALIDATION COMPLETE")
    logger.info(f"{'=' * 60}")

    n_success = len([r for r in case_results if r["status"] == "success"])
    n_failed = len([r for r in case_results if r["status"] == "failed"])

    logger.info(f"Total: {len(case_results)} | Success: {n_success} | Failed: {n_failed}")
    logger.info(f"Results saved to: {args.summary_dir}")

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
