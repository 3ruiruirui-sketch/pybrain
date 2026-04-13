import sys
import json
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_pipeline import run_pipeline, C

results_dir = PROJECT_ROOT / "results"
# Find latest session directory (skip backups; prefer real runs over smoke tests)
sessions = [
    d for d in results_dir.iterdir()
    if d.is_dir() and (d / "session.json").exists() and not d.name.startswith("old_")
]
if not sessions:
    print("No valid sessions found in results directory!")
    sys.exit(1)

prefer = [d for d in sessions if not d.name.lower().startswith("smoke_")]
latest_session_dir = max(prefer or sessions, key=lambda d: d.stat().st_mtime)
session_file = latest_session_dir / "session.json"

print(f"{C.CYAN}Loading autonomous pipeline session from: {session_file}{C.RESET}")

with open(session_file, "r") as f:
    sess = json.load(f)

# Enable ALL stages for a full pipeline run
all_stages = ["stage_1_dicom", "stage_1b_prep", "stage_2_ct", "stage_3_segment", "stage_6_location", "stage_7_morphology", "stage_8_radiomics", "stage_8b_brainiac", "stage_9_report", "stage_9b_report_pt", "stage_10_viz", "stage_11_mricrogl", "stage_12_brats"]
for s in all_stages:
    sess["stages"][s] = True

# Run the pipeline without the wizard
run_pipeline(sess)
