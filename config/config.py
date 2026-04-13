"""
PY-BRAIN — Centralised Configuration
======================================
All paths, thresholds, and constants in one place.
Edit ONLY the PATIENT block for each new case.
All other paths are relative to PROJECT_ROOT and auto-resolve.

Project root: ~/documents/PY-BRAIN/
"""

from pathlib import Path
import torch

# ─────────────────────────────────────────────────────────────────────────
# PROJECT ROOT — auto-detected from this file's location
# ~/documents/PY-BRAIN/config/config.py  →  PROJECT_ROOT = ~/documents/PY-BRAIN
# ─────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────────────────
# ⚙️  PATIENT — update for each new case
# ─────────────────────────────────────────────────────────────────────────
PATIENT = {
    "name":                  "Maria Celeste Coelho Correia Soares",
    "dob":                   "17-02-1945",
    "age":                   "81",
    "ref":                   "HPA630822",
    "exam_date":             "26-02-2026",
    "institution":           "Hospital Particular do Algarve (Faro - Gambelas)",
    "radiologist":           "Dr. Ricardo Cruz Martins / Bruno Moreira (O.M. 42231)",
    "exam_type":             "RM Cranio-Encefalico + TC Cranio",
    "radiologist_volume_cc": 32.0,           # reported lesion volume for validation
    "radiologist_size_cm":   "4.3 x 3.0 x 2.5",
}

# ─────────────────────────────────────────────────────────────────────────
# RAW INPUT — DICOM files from CD
#
# Directory layout inside PY-BRAIN/:
#   Rm_Cranio/          ← MRI DICOM folders (from list_files.txt)
#   tc_Cranio/          ← CT  DICOM folders (from tc_files_list.txt)
# ─────────────────────────────────────────────────────────────────────────
DICOM_MRI_DIR = PROJECT_ROOT / "Rm_Cranio"
DICOM_CT_DIR  = PROJECT_ROOT / "tc_Cranio"

# ─────────────────────────────────────────────────────────────────────────
# CONVERTED NIfTI — output of Stage 1 (dicom_to_nifti.py)
#
#   nifti/monai_ready/       ← T1, T1c, T2, FLAIR  (BraTS inputs)
#   nifti/extra_sequences/   ← DWI, ADC, T2*, CT   (supplementary)
# ─────────────────────────────────────────────────────────────────────────
NIFTI_DIR  = PROJECT_ROOT / "nifti"
MONAI_DIR  = NIFTI_DIR / "monai_ready"
EXTRA_DIR  = NIFTI_DIR / "extra_sequences"

# ─────────────────────────────────────────────────────────────────────────
# MODEL BUNDLE — downloaded once on first run (~500 MB)
# ─────────────────────────────────────────────────────────────────────────
BUNDLE_DIR = PROJECT_ROOT / "models" / "brats_bundle"

# ─────────────────────────────────────────────────────────────────────────
# RESULTS — one timestamped subfolder per run
#   results/
#   ├── ground_truth.nii.gz      ← manual correction (external editor)
#   └── results_TIMESTAMP/       ← outputs of each pipeline run
# ─────────────────────────────────────────────────────────────────────────
RESULTS_DIR  = PROJECT_ROOT / "results"
GROUND_TRUTH = RESULTS_DIR / "ground_truth.nii.gz"

# ─────────────────────────────────────────────────────────────────────────
# DEVICE — Apple Silicon M-series safe defaults
#   MPS  : data tensors (fast)
#   CPU  : model inference (ConvTranspose3D not yet supported on MPS)
# ─────────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE       = torch.device("mps")
    MODEL_DEVICE = torch.device("cpu")
elif torch.cuda.is_available():
    DEVICE = MODEL_DEVICE = torch.device("cuda")
else:
    DEVICE = MODEL_DEVICE = torch.device("cpu")

# ─────────────────────────────────────────────────────────────────────────
# SEGMENTATION THRESHOLDS (tuned for this case — see threshold sweep output)
# ─────────────────────────────────────────────────────────────────────────
WT_THRESH = 0.20    # whole tumour  (more permissive to capture full extent)
TC_THRESH = 0.20    # tumour core   (include necrotic + enhancing)
ET_THRESH = 0.20    # enhancing tumour (more sensitive)

# Sliding window inference parameters
SW_ROI_SIZE   = (96, 96, 96)
SW_BATCH_SIZE = 1
SW_OVERLAP    = 0.5

# ─────────────────────────────────────────────────────────────────────────
# CT HOUNSFIELD UNIT THRESHOLDS (standard radiology values)
# ─────────────────────────────────────────────────────────────────────────
HU_CALCIFICATION_LOW  = 130     # definitive calcification
HU_CALCIFICATION_HIGH = 1000    # upper bound (exclude metal artefacts)
HU_HAEMORRHAGE_LOW    = 50      # acute haemorrhage
HU_HAEMORRHAGE_HIGH   = 90      # upper bound before calcification
HU_TUMOUR_LOW         = 25      # hyperdense tumour tissue
HU_TUMOUR_HIGH        = 60      # upper bound for cellular tumour

# ─────────────────────────────────────────────────────────────────────────
# LABEL COLOURS (BraTS convention)
# ─────────────────────────────────────────────────────────────────────────
LABEL_NAMES  = {1: "Necrotic core", 2: "Edema", 3: "Enhancing tumor"}
LABEL_COLORS = {1: "Blues",         2: "Greens", 3: "Reds"}
LABEL_HEX    = {1: "#4488ff",       2: "#44cc44", 3: "#ff4444"}
LABEL_RGB    = {
    1: [0.27, 0.53, 1.00],   # blue   — necrotic
    2: [0.27, 0.87, 0.27],   # green  — edema
    3: [1.00, 0.27, 0.27],   # red    — enhancing
}

# ─────────────────────────────────────────────────────────────────────────
# MRI SEQUENCE MAP  (DICOM folder name → output NIfTI filename + role)
# Source: list_files.txt from Rm_Cranio/
# ─────────────────────────────────────────────────────────────────────────
MRI_SEQUENCE_MAP = {
    # BRATS sequences — required for AI segmentation
    "t1_mprage_sag_p2_iso_10_2":         ("t1.nii.gz",            "BRATS"),
    "t1_se_tra_civ_13":                  ("t1c.nii.gz",           "BRATS"),
    "pd+t2_tse_tra_7":                   ("t2.nii.gz",            "BRATS"),
    "t2_tirm_tra_darkfluid_320_12":      ("flair.nii.gz",         "BRATS"),
    # Extra sequences — supplementary analysis
    "t2_fl2d_tra_hemo_11":               ("t2star.nii.gz",        "EXTRA"),
    "ep2d_diff_3scan_trace_p2_TRACEW_8": ("dwi.nii.gz",           "EXTRA"),
    "ep2d_diff_3scan_trace_p2_ADC_9":    ("adc.nii.gz",           "EXTRA"),
    "t1_se_cor_civ_14":                  ("t1c_coronal.nii.gz",   "EXTRA"),
    "t1_se_sag_civ_15":                  ("t1c_sagittal.nii.gz",  "EXTRA"),
    "t2_tse_cor_30mm_10":               ("t2_coronal.nii.gz",    "EXTRA"),
    # Localiser/variants — lower priority
    "AX_T1_3":                           ("t1_ax_3.nii.gz",       "EXTRA"),
    "AX_T1_5":                           ("t1_ax_5.nii.gz",       "EXTRA"),
    "COR_T1_4":                          ("t1_cor_4.nii.gz",      "EXTRA"),
    "SAG_T1_6":                          ("t1_sag_6.nii.gz",      "EXTRA"),
}

# ─────────────────────────────────────────────────────────────────────────
# CT SEQUENCE MAP  (DICOM folder name → output NIfTI filename + role)
# Source: tc_files_list.txt from tc_Cranio/
# Ignored: Localizers_1 (scout), Dose_Report_999 (dose report)
# ─────────────────────────────────────────────────────────────────────────
CT_SEQUENCE_MAP = {
    "Cranio_STD_25mm_2": ("ct_brain.nii.gz", "CT"),   # main brain CT
    "OSSO_3":            ("ct_bone.nii.gz",  "CT"),   # bone window — best for calcifications
}

CT_IGNORE = {"Localizers_1", "Dose_Report_999"}       # skip these folders

# ─────────────────────────────────────────────────────────────────────────
# PATHS SUMMARY — printed at startup for verification
# ─────────────────────────────────────────────────────────────────────────
def print_paths():
    """Print all configured paths and whether they exist."""
    paths = {
        "PROJECT_ROOT":   PROJECT_ROOT,
        "DICOM_MRI_DIR":  DICOM_MRI_DIR,
        "DICOM_CT_DIR":   DICOM_CT_DIR,
        "NIFTI_DIR":      NIFTI_DIR,
        "MONAI_DIR":      MONAI_DIR,
        "EXTRA_DIR":      EXTRA_DIR,
        "BUNDLE_DIR":     BUNDLE_DIR,
        "RESULTS_DIR":    RESULTS_DIR,
        "GROUND_TRUTH":   GROUND_TRUTH,
    }
    print("\n  Configured paths:")
    for name, path in paths.items():
        exists = "✅" if path.exists() else "⚠️  (not yet created)"
        print(f"  {name:20s}: {path}  {exists}")

if __name__ == "__main__":
    print_paths()
