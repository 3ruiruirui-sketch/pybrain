#!/usr/bin/env python3
"""
PY-BRAIN — Interactive Pipeline Launchery
==========================================
Run this ONE script to start a full brain tumour analysis.
It will ask where the DICOM files are, collect patient info,
then run the full pipeline automatically.

Usage:
    python3 run_pipeline.py

No config file editing needed.
"""

import sys
import os
import json
import subprocess
import threading
from pathlib import Path
from datetime import datetime

try:
    import pydicom  # type: ignore
except ImportError:
    print("\n  \033[91m❌ Missing requirement: pydicom\033[0m")
    print("  Run:  pip install pydicom\n")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m";  BOLD   = "\033[1m"
    CYAN   = "\033[96m"; GREEN  = "\033[92m"
    YELLOW = "\033[93m"; RED    = "\033[91m"
    BLUE   = "\033[94m"; GREY   = "\033[90m"
    WHITE  = "\033[97m"

def header(text):
    w = 65
    print(f"\n{C.CYAN}{'═'*w}{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}  {text}{C.RESET}")
    print(f"{C.CYAN}{'═'*w}{C.RESET}")

def ok(t):    print(f"  {C.GREEN}✅{C.RESET} {t}")
def info(t):  print(f"  {C.BLUE}ℹ{C.RESET}  {t}")
def warn(t):  print(f"  {C.YELLOW}⚠️ {C.RESET} {t}")
def err(t):   print(f"  {C.RED}❌{C.RESET} {t}")
def step(n,t):print(f"\n{C.BOLD}{C.WHITE}[{n}] {t}{C.RESET}")

def ask(prompt, default="", required=True):
    sfx = f" [{C.GREY}{default}{C.RESET}]" if default else ""
    while True:
        val = input(f"  {C.CYAN}›{C.RESET} {prompt}{sfx}: ").strip()
        if not val and default:
            return default
        if val or not required:
            return val
        warn("This field is required.")

def ask_yn(prompt, default="y"):
    opts = f"{C.GREEN}Y{C.RESET}/n" if default == "y" else f"y/{C.GREEN}N{C.RESET}"
    if not sys.stdin.isatty():
        return default == "y"
    try:
        val = input(f"  {C.CYAN}›{C.RESET} {prompt} [{opts}]: ").strip().lower()
    except (OSError, EOFError):
        # Broken pipe or closed stdin (e.g. auto_run.py, IDE runner, redirected
        # input).  isatty() returned True but read failed — treat as non-interactive
        # and honour the default answer silently.
        return default == "y"
    return (default == "y") if not val else val in ("y", "yes")

def ask_path(prompt: str, must_exist: bool = True) -> Path: # type: ignore
    while True:
        raw  = input(f"  {C.CYAN}›{C.RESET} {prompt}\n    {C.GREY}(drag folder from Finder or type path){C.RESET}\n    Path: ").strip()
        # remove quotes that Finder drag sometimes adds
        raw  = raw.strip("'\"")
        path = Path(os.path.expandvars(os.path.expanduser(raw)))
        if must_exist and not path.exists():
            err(f"Not found: {path}")
            info("Tip: drag the folder from Finder into this Terminal window")
            continue
        if must_exist and not path.is_dir():
            err("That is a file, not a folder. Please provide a folder path.")
            continue
        return path

def pick(prompt: str, options: list, allow_none: bool = False):
    print(f"\n  {C.BOLD}{prompt}{C.RESET}")
    for i, o in enumerate(options, 1):
        print(f"    {C.CYAN}{i}{C.RESET}. {o}")
    if allow_none:
        print(f"    {C.CYAN}0{C.RESET}. Skip")
    while True:
        try:
            n = int(input(f"  {C.CYAN}›{C.RESET} Number: ").strip())
            if allow_none and n == 0:
                return None
            if 1 <= n <= len(options):
                return options[n-1]
        except ValueError:
            pass
        warn(f"Enter a number between {'0' if allow_none else '1'} and {len(options)}")


# ─────────────────────────────────────────────────────────────────────────
# DICOM SCANNER
# ─────────────────────────────────────────────────────────────────────────

def _guess_type(name):
    n = name.lower()
    # IGNORE first
    if any(x in n for x in [
        "local","scout","topogram","dose","report","999",
        "backup","log","thumb","preview"]):
        return "IGNORE"
    # T1c BEFORE T1 (civ = contrast IV)
    if any(x in n for x in [
        "civ","_civ","contrast","gad","post","t1c","t1+","+c",
        "t1_ce","t1ce","t1_gd"]):
        return "T1c"
    # T1 3D
    if any(x in n for x in [
        "mprage","mp2rage","t1_mprage","t1_3d","t1_sag","3d_t1",
        "ax_t1","t1_ax","cor_t1","sag_t1","t1_se","t1_tse","t1w"]):
        return "T1"
    # FLAIR
    if any(x in n for x in [
        "flair","tirm","darkfluid","dark_fluid","dark-fluid",
        "t2_flair","fse_flair","t2flair"]):
        return "FLAIR"
    # ADC before DWI
    if any(x in n for x in ["adc","apparent_diff","_adc_"]) or n.endswith("adc"):
        return "ADC"
    # DWI
    if any(x in n for x in [
        "tracew","trace","dwi","diff","diffusion","ep2d_diff","dti"]):
        return "DWI"
    # T2*
    if any(x in n for x in [
        "hemo","t2star","t2_star","fl2d","t2*","gre","swi","haemo"]):
        return "T2star"
    # T2 (after flair/t2star)
    if any(x in n for x in [
        "pd+t2","pd_t2","t2_tse","t2_fse","t2_tra","t2_cor",
        "t2_sag","t2w","tse_t2","t2"]):
        return "T2"
    # CT bone
    if any(x in n for x in ["osso","bone","osseo"]):
        return "CT_bone"
    # CT brain
    if any(x in n for x in ["cranio","std","ct_brain","encefal"]):
        return "CT_brain"
    return "UNKNOWN"


# Known container folder names — these hold series but are not series themselves
_CONTAINERS = {
    "rm_cranio","rm cranio","rmcranio",
    "tc_cranio","tc cranio","tccranio",
    "dicom","dcm","images","img","data","raw","scan","cd","cdrom",
    "disc","disk","root","backup","backups","patient","study","series",
    "cache","thumb","preview",
}

# Directories that are NEVER DICOM — skip entirely during scanning
_IGNORE_DIRS = {
    ".venv", "venv", ".env", "env", "__pycache__", ".git", ".svn",
    "node_modules", ".tox", ".mypy_cache", ".pytest_cache",
    "models", "results", "scripts", "config", ".agents", ".gemini",
    "nifti", "monai_ready", "extra_sequences", "brats_bundle",
    "site-packages", "dist-info", "lib", "bin", "include",
}

def _should_skip_dir(path: Path) -> bool:
    """Check if any component of the path is in _IGNORE_DIRS."""
    for part in path.parts:
        if part.lower() in _IGNORE_DIRS or part.endswith(".dist-info"):
            return True
        # Skip hidden directories (except the root might start with .)
        if part.startswith(".") and part not in (".", ".."):
            return True
    return False

def scan_folder(root: Path) -> dict:
    """
    Recursively scan ANY folder structure and return every DICOM series.
    Uses rglob to walk all levels. Groups files by immediate parent folder.
    Skips known container names (Rm_Cranio, Tc_Cranio, DICOM, etc.).
    Skips non-DICOM directories (.venv, scripts, models, results, etc.).
    """
    result = {}
    if not root or not root.exists():
        return result

    # Collect all image files recursively, skipping non-DICOM dirs
    all_files = []
    for f in root.rglob("*"):
        if not f.is_file() or f.name.startswith("."):
            continue
        # Skip files inside ignored directory trees
        try:
            rel = f.relative_to(root)
        except ValueError:
            continue
        if _should_skip_dir(rel):
            continue
        ext = f.suffix.lower()
        if ext in (".dcm", ".ima", ".img", "") :
            all_files.append(f)

    if not all_files:
        # Fallback: any file (still excluding ignored dirs)
        all_files = []
        for f in root.rglob("*"):
            if not f.is_file() or f.name.startswith("."):
                continue
            try:
                rel = f.relative_to(root)
            except ValueError:
                continue
            if _should_skip_dir(rel):
                continue
            all_files.append(f)

    # Group by immediate parent
    import typing
    from collections import defaultdict
    groups: typing.DefaultDict[Path, list[Path]] = defaultdict(list) # type: ignore
    for f in all_files:
        groups[f.parent].append(f)

    for folder_path, files in sorted(groups.items()):
        fname = folder_path.name
        # Skip root itself
        if folder_path == root:
            continue
        # Skip known container names
        if fname.lower() in _CONTAINERS:
            continue
        n_files  = len(files)
        seq_type = _guess_type(fname)
        # Skip IGNORE folders with very few files
        if seq_type == "IGNORE" and n_files < 5:
            continue
        result[fname] = {
            "path":    folder_path,
            "n_files": n_files,
            "type":    seq_type,
        }

    return result

ROLE_FROM_TYPE = {
    "T1":"T1", "T1c":"T1c", "T2":"T2", "FLAIR":"FLAIR",
    "DWI":"DWI", "ADC":"ADC", "T2star":"T2star",
    "CT_brain":"CT", "CT_bone":"CT_bone",
}

REQUIRED = {"T1":"T1 pre-contrast (3D MPRAGE)",
            "T1c":"T1 post-contrast (after IV contrast)",
            "T2":"T2 axial",
            "FLAIR":"T2 FLAIR / TIRM Dark-Fluid"}

OPTIONAL = {"DWI":"Diffusion Weighted Imaging",
            "ADC":"ADC map",
            "T2star":"T2* haemosiderin/calcification",
            "CT":"CT brain (soft tissue window)",
            "CT_bone":"CT bone window"}

def auto_assign(series: dict) -> dict[str, str]:
    """
    Assign series to roles using priority scoring so the best
    quality folder always wins when multiple candidates exist.

    Priority rules:
      T1    → prefer mprage/3d/iso (high-res 3D) over ax/cor/sag variants
      T1c   → prefer axial (_tra_) over coronal (_cor_) or sagittal (_sag_)
      T2    → prefer axial tse/fse over coronal thick-slice
      FLAIR → any tirm/flair (usually only one)
    """

    # Score keywords — higher = better candidate for that role
    T1_PREFER    = ["mprage","mp2rage","iso","3d","t1_3d"]
    T1_AVOID     = ["ax_t1","cor_t1","sag_t1","ax_","cor_","sag_",
                    "_3","_4","_5","_6"]   # numbered axial variants
    T1C_PREFER   = ["_tra_","tra","axial","ax"]
    T1C_AVOID    = ["_cor_","_sag_","coronal","sagittal"]
    T2_PREFER    = ["_tra_","tra","tse","fse","axial"]
    T2_AVOID     = ["_cor_","cor","30mm","thick"]

    def score(fname: str, prefer: list[str], avoid: list[str]) -> int:
        n = fname.lower()
        s: int = 0
        for p in prefer:
            if p in n:
                s += 10 # type: ignore
        for a in avoid:
            if a in n:
                s -= 5 # type: ignore
        # More files generally = higher resolution = better
        n_files = int(series.get(fname, {}).get("n_files", 0))
        s += n_files // 20 # type: ignore
        return s

    # Group candidates by role
    import typing
    candidates: dict[str, list[str]] = {}
    for fname, d in series.items():
        role = ROLE_FROM_TYPE.get(d["type"])
        if role:
            candidates.setdefault(role, []).append(fname)

    out  = {}
    used = set()

    for role, fnames in candidates.items():
        if not fnames:
            continue
        # Pick best candidate using role-specific scoring
        if role == "T1":
            best = max(fnames, key=lambda f: score(f, T1_PREFER, T1_AVOID))
        elif role == "T1c":
            best = max(fnames, key=lambda f: score(f, T1C_PREFER, T1C_AVOID))
        elif role == "T2":
            best = max(fnames, key=lambda f: score(f, T2_PREFER, T2_AVOID))
        else:
            # For other roles just take the one with most files
            best = max(fnames, key=lambda f: series[f]["n_files"])

        if best not in used:
            out[role] = best
            used.add(best)

    return out


# ─────────────────────────────────────────────────────────────────────────
# SESSION
# ─────────────────────────────────────────────────────────────────────────

def save_session(sess, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / "session.json"
    def _s(o):
        if isinstance(o, Path): return str(o)
        if isinstance(o, dict): return {k:_s(v) for k,v in o.items()}
        if isinstance(o, list): return [_s(v) for v in o]
        return o
    with open(path,"w") as f:
        json.dump(_s(sess), f, indent=2)
    return path


# ─────────────────────────────────────────────────────────────────────────
# PATIENT INFO EXTRACTION
# ─────────────────────────────────────────────────────────────────────────

def extract_patient_info_from_dicom(dicom_dir):
    """Scan for the first valid DICOM file and extract patient details."""
    dicom_path = Path(dicom_dir)
    # Search for first .dcm or .IMA
    first_file = None
    for ext in ['**/*.dcm', '**/*.IMA']:
        try:
            first_file = next(dicom_path.glob(ext))
            break
        except StopIteration:
            continue
            
    if not first_file:
        # Fallback to any file if extensions are missing
        for f in dicom_path.rglob('*'):
            if f.is_file() and not f.name.startswith('.') and not _should_skip_dir(f.relative_to(dicom_path)):
                first_file = f
                break

    if not first_file:
        return {"name": "Unknown", "age": "Unknown", "sex": "Unknown"}

    try:
        ds = pydicom.dcmread(str(first_file), stop_before_pixels=True)
        raw_name = str(getattr(ds, 'PatientName', 'Unknown'))
        clean_name = raw_name.replace('^', ' ').strip() if raw_name != 'Unknown' else 'Unknown'
        
        raw_age = str(getattr(ds, 'PatientAge', 'Unknown'))
        clean_age = raw_age.replace('Y', '').strip() if raw_age != 'Unknown' else 'Unknown'
        
        clean_sex = str(getattr(ds, 'PatientSex', 'Unknown')).strip()
        
        return {
            "name": clean_name or "Unknown",
            "age": clean_age or "Unknown",
            "sex": clean_sex or "Unknown"
        }
    except Exception:
        return {"name": "Unknown", "age": "Unknown", "sex": "Unknown"}

# ─────────────────────────────────────────────────────────────────────────
# WIZARD
# ─────────────────────────────────────────────────────────────────────────

def wizard():

    print(f"""
{C.CYAN}{'═'*65}
{C.BOLD}  PY-BRAIN  —  Brain Tumour AI Analysis Pipeline
{C.RESET}{C.CYAN}  MONAI BraTS SegResNet  |  Research Use Only
{'═'*65}{C.RESET}
""")

    import typing
    sess: dict[str, typing.Any] = {"created": datetime.now().isoformat(),
            "project_root": str(PROJECT_ROOT)}

    # ── Previous session? ─────────────────────────────────────────────
    # Search in PROJECT_ROOT and also common alternative locations
    search_roots = [
        PROJECT_ROOT,
        Path.home() / "documents" / "PY-BRAIN",
        Path.home() / "Downloads"  / "PY-BRAIN",
        Path.home() / "Documents"  / "PY-BRAIN",
    ]
    prev = []
    for root in search_roots:
        prev += sorted(root.glob("results/*/session.json")) \
                if root.exists() else []
    prev = sorted(set(prev))
    if prev:
        info(f"Found {len(prev)} previous session(s).")
        if ask_yn("Load a previous session?", default="n"):
            names = [f"{p.parent.name} ({p.parent.parent.parent.name})" for p in prev]
            ch = pick("Select session:", names)
            if ch:
                # Get the index of the selected name
                idx = names.index(ch)
                with open(prev[idx]) as f:
                    sess = json.load(f)
                ok(f"Loaded: {ch}")
                return sess

    # ─────────────────────────────────────────────────────────────────
    header("STEP 1 — DICOM FILES LOCATION")

    print(f"""
  {C.BOLD}What are DICOM files?{C.RESET}
  DICOM (.dcm) files are the raw medical images on the CD-ROM
  provided by the hospital after an MRI or CT scan.

  {C.BOLD}What path to enter:{C.RESET}
  Enter the folder that contains your DICOM series sub-folders.
  The pipeline will scan it recursively for all MRI and CT scans.

  Example — if your folders look like this:
    {C.CYAN}DICOM/
    ├── Rm_Cranio/
    │   ├── t1_mprage_.../
    │   └── flair_.../
    └── Tc_Cranio/
        └── Cranio_STD_.../{C.RESET}

  Then enter: {C.CYAN}DICOM{C.RESET} (the top-level parent folder).

  {C.GREY}Tip: Drag the folder from Finder into this Terminal window{C.RESET}
""")

    step("1a","DICOM study folder")
    dicom_dir = ask_path("Path to DICOM root folder:")

    sess["mri_dicom_dir"] = str(dicom_dir)
    sess["ct_dicom_dir"]  = str(dicom_dir)

    # ── Scan ──────────────────────────────────────────────────────────
    step("1b","Scanning DICOM folder recursively…")
    all_series: dict[str, dict] = scan_folder(dicom_dir)

    if not all_series:
        err("No series found. Make sure sub-folders with image files exist.")
        sys.exit(1)

    mri_count = sum(1 for d in all_series.values() if d["type"] not in ("IGNORE", "UNKNOWN", "CT_brain", "CT_bone"))
    ct_count  = sum(1 for d in all_series.values() if d["type"] in ("CT_brain", "CT_bone"))

    ok(f"Found {mri_count} MRI + {ct_count} CT series")
    print()
    print(f"  {'Folder':<38}  {'Files':>5}  {'Detected type'}")
    print(f"  {'─'*38}  {'─'*5}  {'─'*18}")
    for nm, d in all_series.items():
        col = (C.GREEN  if d["type"] not in ("IGNORE","UNKNOWN") else
               C.GREY   if d["type"] == "IGNORE" else C.YELLOW)
        print(f"  {nm[:38]:<38}  {d['n_files']:>5}  {col}{d['type']}{C.RESET}")

    # ─────────────────────────────────────────────────────────────────
    header("STEP 2 — PATIENT INFORMATION")
    info("Extracting automatically from DICOM headers...")
    
    p = extract_patient_info_from_dicom(dicom_dir)
    sess['patient'] = p
    print(f"  {C.GREEN}✅ Auto-detected Patient: {p['name']} ({p['age']}y, {p['sex']}){C.RESET}")
    input(f"\n  {C.CYAN}›{C.RESET} Press Enter to continue...")

    # ─────────────────────────────────────────────────────────────────
    header("STEP 3 — CONFIRM SEQUENCE ASSIGNMENTS")
    info("PY-BRAIN auto-detected which folder is which sequence.")
    info("Please verify — correct any wrong assignments.\n")

    asgn: dict[str, str] = auto_assign(all_series)

    print(f"  {'Role':<10}  {'Assigned folder':<40}  Status")
    print(f"  {'─'*10}  {'─'*40}  {'─'*10}")
    for role, desc in {**REQUIRED, **OPTIONAL}.items():
        folder = str(asgn.get(role,""))
        n      = all_series.get(folder, {}).get("n_files", 0) if folder else 0
        if folder:
            status = f"{C.GREEN}✅ auto-matched{C.RESET}"
        elif role in REQUIRED:
            status = f"{C.RED}❌ REQUIRED — not found{C.RESET}"
        else:
            status = f"{C.GREY}── optional{C.RESET}"
        folder_str = folder[:40] # type: ignore
        print(f"  {role:<10}  {folder_str:<40}  {status}")
        if folder:
            print(f"  {'':10}  {C.GREY}{desc}  ({n} files){C.RESET}")

    print()
    if not ask_yn("Are these assignments correct?", default="y"):
        folder_names = list(all_series.keys())
        for role in list(REQUIRED.keys()) + list(OPTIONAL.keys()):
            desc    = REQUIRED.get(role, OPTIONAL.get(role,""))
            current = asgn.get(role,"none")
            print(f"\n  {C.BOLD}{role}{C.RESET} — {desc}")
            print(f"  Current: {C.CYAN}{current}{C.RESET}")
            opts   = ["(keep current)", "(clear/skip)"] + folder_names
            choice = str(pick("Select folder:", opts))
            if choice == "(clear/skip)":
                asgn.pop(role, None)
            elif choice != "(keep current)":
                asgn[role] = choice

    missing = [r for r in REQUIRED if r not in asgn]
    if missing:
        warn(f"Missing required sequences: {missing}")
        warn("AI will use heuristic fallback (less accurate).")
        if not ask_yn("Continue anyway?", default="y"):
            sys.exit(0)

    sess["series_paths"]  = {nm: str(d["path"])
                              for nm, d in all_series.items()}
    sess["assignments"]   = asgn

    # ─────────────────────────────────────────────────────────────────
    header("STEP 4 — OUTPUT FOLDER")

    default_out = PROJECT_ROOT / "results"
    print(f"  Default: {C.CYAN}{default_out}{C.RESET}")
    if ask_yn("Use a different results folder?", default="n"):
        results_dir = ask_path("Results folder path:", must_exist=False)
    else:
        results_dir = default_out

    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize name - remove quotes, paths, and weird chars to prevent path injection
    name_raw  = str(p.get("name", "patient"))
    import re
    id_safe   = re.sub(r'[^a-zA-Z0-9_\-]', '_', name_raw).strip('_')
    safe_base: str = id_safe if id_safe else "patient"
    safe           = safe_base[:20] # type: ignore
    out_dir   = results_dir / f"{safe}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sess["results_dir"]  = str(results_dir)
    sess["output_dir"]   = str(out_dir)
    sess["ground_truth"] = str(results_dir / "ground_truth.nii.gz")
    sess["monai_dir"]    = str(out_dir / "nifti" / "monai_ready")
    sess["extra_dir"]    = str(out_dir / "nifti" / "extra_sequences")
    sess["nifti_dir"]    = str(out_dir / "nifti")
    sess["bundle_dir"]   = str(PROJECT_ROOT / "models" / "brats_bundle")

    for d in [sess["monai_dir"], sess["extra_dir"], sess["bundle_dir"]]:
        Path(d).mkdir(parents=True, exist_ok=True)

    ok(f"Output: {out_dir}")

    # ── Show actual folder structure using real paths from session ────────
    dicom_rel = Path(str(dicom_dir)).relative_to(PROJECT_ROOT) \
                if str(dicom_dir).startswith(str(PROJECT_ROOT)) \
                else Path(str(dicom_dir))
    out_rel = out_dir.relative_to(PROJECT_ROOT) \
              if str(out_dir).startswith(str(PROJECT_ROOT)) \
              else out_dir

    # List top assigned series
    top_series = []
    for i, (k, v) in enumerate(asgn.items()):
        if i < 4: top_series.append((k, v))
    series_lines = "\n".join(
        f"  │   ├── {folder}/"
        for role, folder in top_series
        if role in ("T1","T1c","T2","FLAIR")
    )

    print(f"""
  {C.BOLD}Project folder structure{C.RESET}  ({C.GREY}your project is at:{C.RESET} {C.CYAN}{PROJECT_ROOT}{C.RESET})

  {PROJECT_ROOT.name}/
  ├── {C.CYAN}{dicom_rel}/{C.RESET}         ← DICOM source folder
{series_lines}
  │   └── ... ({len(all_series)} series total)
  ├── {C.GREEN}models/brats_bundle/{C.RESET}    ← AI model (auto-downloaded ~500 MB)
  ├── {C.GREEN}results/{C.RESET}                ← all patient outputs
  │   ├── {C.YELLOW}ground_truth.nii.gz{C.RESET} ← manual correction (external editor)
  │   └── {C.GREEN}{out_rel}/{C.RESET}
  │       ├── {C.CYAN}nifti/monai_ready/{C.RESET}   ← T1, T1c, T2, FLAIR (Stage 1)
  │       ├── {C.CYAN}nifti/extra_sequences/{C.RESET}← DWI, ADC, T2*, CT  (Stage 1)
  │       ├── segmentation_full.nii.gz  (Stage 3)
  │       ├── interactive_viewer.html   (Stage 3)
  │       ├── tumor_stats.json          (Stage 3)
  │       ├── validation_metrics.json   (Stage 5)
  │       ├── tumour_location.json       (Stage 6)
  │       ├── morphology.json           (Stage 7)
  │       ├── radiomics_features.json   (Stage 8)
  │       ├── viz_slice_animation.html  (Stage 10)
  │       ├── mricrogl/                  (Stage 11)
  │       ├── brats_figure1_style.png   (Stage 12)
  │       ├── report_TIMESTAMP.pdf       (Stage 9)
  │       └── session.json
  └── {C.CYAN}scripts/{C.RESET}                ← 9 pipeline scripts
""")

    # ─────────────────────────────────────────────────────────────────
    header("STEP 5 — SELECT PIPELINE STAGES")

    stages = {
        "stage_1_dicom":     ask_yn("Stage 1 — Convert DICOM → NIfTI",        "y"),
        "stage_1b_prep":     ask_yn("Stage 1b — BraTS Refinement (RAI + Masking)", "y"),
        "stage_2_ct":        ask_yn("Stage 2 — CT integration",
                                     "y" if ct_count > 0 else "n"),
        "stage_2b_ct_merge": ask_yn("Stage 2b — CT → Segmentation Merge (post-Stage 3)", "n"),
        "stage_3_segment":   ask_yn("Stage 3 — AI tumour segmentation",        "y"),
        "stage_4_review":    ask_yn("Stage 4 — Manual review (open in 3D viewer)", "y"),
        "stage_5_validate":  ask_yn("Stage 5 — Validation metrics",            "y"),
        "stage_6_location":  ask_yn("Stage 6 — Automated Location Analysis",   "y"),
        "stage_7_morphology":ask_yn("Stage 7 — Detailed Morphology Metrics",   "y"),
        "stage_8_radiomics": ask_yn("Stage 8 — Radiomics + ML classification","y"),
        "stage_8b_brainiac": ask_yn("Stage 8b — Genomic Prediction (BrainIAC)","n"),
        "stage_9_report":    ask_yn("Stage 9 — Generate PDF report (English)",  "y"),
        "stage_9b_report_pt":ask_yn("Stage 9b — Generate PDF report (Portuguese)", "n"),
        "stage_10_viz":      ask_yn("Stage 10 — Enhanced Visualisation",       "y"),
        "stage_11_mricrogl": ask_yn("Stage 11 — MRIcroGL Advanced 3D Renders","n"),
        "stage_12_brats":    ask_yn("Stage 12 — BraTS 2021 Style Figure",     "n"),
    }
    sess["stages"] = stages

    # Focus exclusively on our hardened SegResNet-TTA4 engine
    sess["segmentation_script"] = "3_brain_tumor_analysis.py"
    sess["segmentation_model"] = "SegResNet-TTA4"
    info("Using highly optimized SegResNet with TTA4 (Test-Time Augmentation)")

    # ─────────────────────────────────────────────────────────────────
    header("STEP 6 — CONFIRM & START")

    print(f"""
  {C.BOLD}Patient{C.RESET}
    {p.get('name', 'Unknown')}  |  Age {p.get('age', 'Unknown')}  |  Sex {p.get('sex', 'Unknown')}

  {C.BOLD}DICOM source{C.RESET}
    Path: {dicom_dir}
    Info: {mri_count} MRI + {ct_count} CT series found

  {C.BOLD}Sequence assignments{C.RESET}""")
    for role, folder in asgn.items():
        n = all_series.get(folder, {}).get("n_files", "?")
        print(f"    {C.GREEN}{role:<10}{C.RESET} ← {folder}  ({n} files)")

    print(f"\n  {C.BOLD}Output{C.RESET}  {out_dir}")

    # Live tree of actual folders created so far
    print(f"\n  {C.BOLD}Folders created on disk:{C.RESET}")
    def _tree(path: Path, prefix: str = "  ", max_depth: int = 3, depth: int = 0):
        if depth > max_depth or not path.exists():
            return
        items = sorted([x for x in path.iterdir()
                        if not x.name.startswith(".")])
        for i, item in enumerate(items):
            connector = "└── " if i == len(items)-1 else "├── "
            ext       = "/ " if item.is_dir() else "  "
            col       = C.CYAN if item.is_dir() else C.GREY
            print(f"  {prefix}{connector}{col}{item.name}{ext}{C.RESET}")
            if item.is_dir() and depth < max_depth:
                extension = "    " if i == len(items)-1 else "│   "
                _tree(item, prefix + extension, max_depth, depth + 1)

    print(f"  {C.CYAN}{PROJECT_ROOT.name}/{C.RESET}")
    _tree(PROJECT_ROOT, prefix="  ")

    print(f"\n  {C.BOLD}Stages{C.RESET}")
    stage_labels = {
        "stage_1_dicom":     "Stage 1 — DICOM → NIfTI",
        "stage_1b_prep":     "Stage 1b — BraTS Refinement",
        "stage_2_ct":        "Stage 2 — CT Integration",
        "stage_2b_ct_merge": "Stage 2b — CT Merge (post-Stage 3)",
        "stage_3_segment":   "Stage 3 — AI Segmentation",
        "stage_4_review":    "Stage 4 — Manual Review",
        "stage_5_validate":  "Stage 5 — Validation Metrics",
        "stage_6_location":  "Stage 6 — Location Analysis",
        "stage_7_morphology":"Stage 7 — Morphology",
        "stage_8_radiomics": "Stage 8 — Radiomics",
        "stage_8b_brainiac": "Stage 8b — BrainIAC Prediction",
        "stage_9_report":    "Stage 9 — PDF Report (EN)",
        "stage_9b_report_pt":"Stage 9b — PDF Report (PT)",
        "stage_10_viz":      "Stage 10 — Visualisation",
        "stage_11_mricrogl": "Stage 11 — MRIcroGL",
        "stage_12_brats":    "Stage 12 — BraTS Figure",
    }
    for s, en in stages.items():
        icon = f"{C.GREEN}✅{C.RESET}" if en else f"{C.GREY}⏭{C.RESET}"
        label = stage_labels.get(s, s.replace('_', ' '))
        print(f"    {icon}  {label}")

    print()
    if not ask_yn(f"{C.BOLD}{C.GREEN}Start the pipeline now?{C.RESET}", "y"):
        path = save_session(sess, out_dir)
        info(f"Session saved → {path}")
        info("Run again to start.")
        sys.exit(0)

    path = save_session(sess, out_dir)
    ok(f"Session saved → {path}")
    return sess


# ─────────────────────────────────────────────────────────────────────────
# PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────────────────

def run_stage(label, script_name, sess, extra_env=None, required=False):
    scripts_dir = PROJECT_ROOT / "scripts"
    script_path = scripts_dir / script_name

    if not script_path.exists():
        warn(f"Script not found: {script_name} — skipping")
        return False

    header(f"RUNNING  {label}")
    env = os.environ.copy()
    env["PYBRAIN_SESSION"] = str(Path(sess["output_dir"]) / "session.json")
    if extra_env:
        env.update(extra_env)

    # Stream output live so the user sees progress during long stages
    # (Stage 3 inference can take 5–30 min on CPU — silent terminal is confusing).
    # stderr is also streamed live; we collect it separately only to show on failure.
    stderr_lines: list = []
    proc = subprocess.Popen(
        [sys.executable, "-u", str(script_path)],   # -u = unbuffered stdout
        env=env,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    # Stream stdout live; collect stderr for failure diagnostics
    def _drain_stderr():
        for line in proc.stderr:  # type: ignore[union-attr]
            stderr_lines.append(line)
            sys.stderr.write(line)
            sys.stderr.flush()
    t = threading.Thread(target=_drain_stderr, daemon=True)
    t.start()
    for line in proc.stdout:    # type: ignore[union-attr]
        sys.stdout.write(line)
        sys.stdout.flush()
    proc.wait()
    t.join()

    if proc.returncode == 0:
        ok(f"{label} — done")
        return True
    else:
        err(f"{label} — failed (exit {proc.returncode})")
        if stderr_lines:
            print("".join(stderr_lines[-40:]))  # show last 40 stderr lines
        if required:
            err(f"{label} is required — aborting pipeline.")
            sys.exit(1)
        if not ask_yn("Continue with next stage?", default="n"):
            print(f"\n  {C.YELLOW}Pipeline halted by user.{C.RESET}\n")
            sys.exit(0)
        return False


def run_stage_review(sess):
    """Stage 4 — prompt user to open segmentation in a 3D viewer for manual review."""
    out_dir = Path(sess["output_dir"])
    seg_path = out_dir / "segmentation_full.nii.gz"
    interactive = out_dir / "interactive_viewer.html"
    html_path = out_dir / "viz_slice_animation.html"

    print(f"""
  {C.BOLD}📋 Manual Review — Stage 4{C.RESET}

  Please open the segmentation in a 3D viewer to verify quality.
  If corrections are needed, edit the mask and save as:
    {C.YELLOW}{sess.get('ground_truth', out_dir / 'ground_truth.nii.gz')}{C.RESET}

  Files to review:
    • {seg_path}
    • {interactive}
    • {html_path}
""")
    if interactive.exists():
        info(f"Interactive viewer: file://{interactive}")
    if html_path.exists():
        info(f"Slice animation:     file://{html_path}")

    ok("Manual review complete — segmentation accepted.")
    return True


def find_seg_session(results_base: Path, current_session: str) -> str:
    """Return most recent session dir that has segmentation_ensemble.nii.gz"""
    if not results_base.exists():
        return current_session
    candidates = sorted([
        d for d in results_base.iterdir()
        if d.is_dir()
        and d.name != current_session
        and (d / "segmentation_ensemble.nii.gz").exists()
    ], key=lambda x: x.name, reverse=True)
    if candidates:
        return candidates[0].name
    return current_session


def run_pipeline(sess):
    st = sess.get("stages", {})
    p  = sess.get("patient", {})

    # ─────────────────────────────────────────────────────────────────
    # PREFLIGHT GATE
    # ─────────────────────────────────────────────────────────────────
    print(f"\n{C.BOLD}{C.CYAN}🔍 Running preflight checks...{C.RESET}")
    pf_script = PROJECT_ROOT / "scripts" / "0_preflight_check.py"
    session_json = Path(sess["output_dir"]) / "session.json"
    if not session_json.exists():
        save_session(sess, Path(sess["output_dir"]))
    pf_env = os.environ.copy()
    pf_env["PYBRAIN_SESSION"] = str(session_json)
    pf_stderr: list = []
    pf_proc = subprocess.Popen(
        [sys.executable, "-u", str(pf_script)],
        env=pf_env,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    def _pf_stderr():
        for ln in pf_proc.stderr:  # type: ignore[union-attr]
            pf_stderr.append(ln); sys.stderr.write(ln); sys.stderr.flush()
    threading.Thread(target=_pf_stderr, daemon=True).start()
    for ln in pf_proc.stdout:  # type: ignore[union-attr]
        sys.stdout.write(ln); sys.stdout.flush()
    pf_proc.wait()
    if pf_proc.returncode != 0:
        print(f"\n{C.RED}{C.BOLD}❌ Preflight failed — pipeline aborted.{C.RESET}")
        if pf_stderr:
            print("".join(pf_stderr[-20:]))
        print("   Fix the errors above and re-run.\n")
        sys.exit(1)
    print(f"{C.GREEN}✅ Preflight passed — starting pipeline...{C.RESET}\n")

    print(f"""
{C.CYAN}{'═'*65}
{C.BOLD}  PIPELINE STARTING
{C.RESET}{C.CYAN}  Patient : {p.get('name','?')}
  Output  : {sess['output_dir']}
{'═'*65}{C.RESET}
""")

    results_base = Path(sess.get("results_dir", PROJECT_ROOT / "results"))
    session_name = Path(sess["output_dir"]).name
    seg_session = find_seg_session(results_base, session_name)
    seg_env = {"PYBRAIN_SEG_SESSION": seg_session}

    results = {}
    if st.get("stage_1_dicom"):
        results[1] = run_stage("Stage 1 — DICOM → NIfTI",      "1_dicom_to_nifti.py",   sess)
    if st.get("stage_1b_prep"):
        results["1b"] = run_stage("Stage 1b — BraTS Refinement", "1b_brats_preproc.py", sess)
    # Stage 2 before 3 so registered CT (ct_brain_registered.nii.gz) exists for segmentation boost
    if st.get("stage_2_ct"):
        results[2] = run_stage("Stage 2 — CT Integration",      "2_ct_integration.py",   sess)
    if st.get("stage_3_segment"):
        seg_script = sess.get("segmentation_script", "3_brain_tumor_analysis.py")
        results[3] = run_stage("Stage 3 — AI Segmentation", seg_script, sess, extra_env=seg_env)

    # Stage 2b after Stage 3 — merge CT masks into the fresh segmentation
    if st.get("stage_2b_ct_merge"):
        results["2b"] = run_stage(
            "Stage 2b — CT → Segmentation Merge",
            "2_ct_integration.py",
            sess,
            extra_env={"PYBRAIN_MERGE_ONLY": "1"}
        )

    # ── Stage 4: Manual Review ──────────────────────────────────────────
    if st.get("stage_4_review"):
        results[4] = run_stage_review(sess)

    # ── Stage 5: Validation Metrics ────────────────────────────────────
    if st.get("stage_5_validate"):
        results[5] = run_stage("Stage 5 — Validation Metrics",
                               "5_validate_segmentation.py", sess,
                               extra_env=seg_env, required=False)

    if st.get("stage_6_location"):
        results[6] = run_stage("Stage 6 — Location Analysis",   "6_tumour_location.py",  sess, extra_env=seg_env)
    if st.get("stage_7_morphology"):
        results[7] = run_stage("Stage 7 — Morphology",          "7_tumour_morphology.py",sess, extra_env=seg_env)
    if st.get("stage_8_radiomics"):
        results[8] = run_stage("Stage 8 — Radiomics",           "8_radiomics_analysis.py",sess, extra_env=seg_env)
    if st.get("stage_8b_brainiac"):
        results[81] = run_stage("Stage 8b — Genomic Prediction (BrainIAC)", "8b_brainiac_prediction.py", sess)
    if st.get("stage_10_viz"):
        results[10]= run_stage("Stage 10 — Visualisation",      "10_enhanced_visualisation.py", sess, extra_env=seg_env)
    if st.get("stage_11_mricrogl"):
        results[11] = run_stage("Stage 11 — MRIcroGL Advanced",  "11_mricrogl_visualisation.py", sess, extra_env=seg_env)
    if st.get("stage_9_report"):
        results[9] = run_stage("Stage 9 — PDF Report (English)", "9_generate_report.py",  sess)
    if st.get("stage_9b_report_pt"):
        results[91] = run_stage("Stage 9b — PDF Report (Portuguese)", "9_generate_report_pt.py",  sess)
    if st.get("stage_12_brats"):
        results[12] = run_stage("Stage 12 — BraTS Figure 1", "12_brats_figure1.py", sess)

    header("PIPELINE COMPLETE")
    passed = sum(1 for v in results.values() if v)
    total  = len(results)
    out    = Path(sess['output_dir'])

    print(f"""
  Stages completed : {C.GREEN}{passed}{C.RESET}/{total}
  Output folder    : {C.CYAN}{out}{C.RESET}

  {C.GREY}ℹ️  To free up space, you can delete the 'ct_work' folder 
     and any original NIfTI files in 'nifti/' after Stage 2.{C.RESET}
""")

    # Live tree of output folder
    if out.exists():
        print(f"  {C.BOLD}Files produced:{C.RESET}")
        print(f"  {C.CYAN}{out.name}/{C.RESET}")
        items = sorted([x for x in out.rglob("*")
                        if not x.name.startswith(".")
                        and x.suffix in (".gz",".json",".html",
                                         ".png",".pdf",".txt")])
        # Group by subfolder
        from collections import defaultdict as _dd
        by_dir = _dd(list)
        for item in items:
            rel = item.relative_to(out)
            parts = rel.parts
            subdir = parts[0] if len(parts) > 1 else ""
            by_dir[subdir].append(item)

        for subdir, files in sorted(by_dir.items()):
            if subdir:
                print(f"  ├── {C.CYAN}{subdir}/{C.RESET}")
                for item in sorted(files, key=lambda p: str(p.relative_to(out))):
                    rel_to_subdir = item.relative_to(out / subdir)
                    size = item.stat().st_size / 1024
                    size_s = f"{size/1024:.1f} MB" if size>1024 else f"{size:.0f} KB"
                    print(f"  │   ├── {C.GREY}{str(rel_to_subdir):<40}{C.RESET}  {C.GREY}{size_s}{C.RESET}")
            else:
                for f in sorted(files):
                    size = f.stat().st_size / 1024
                    size_s = f"{size/1024:.1f} MB" if size>1024 else f"{size:.0f} KB"
                    fname = f.name
                    icon = ("📊" if fname.endswith(".json") else
                            "🌐" if fname.endswith(".html") else
                            "🖼 " if fname.endswith(".png")  else
                            "📄" if fname.endswith(".pdf")  else
                            "🧠" if fname.endswith(".gz")   else "  ")
                    print(f"  ├── {icon} {C.GREEN}{fname:<42}{C.RESET}  {C.GREY}{size_s}{C.RESET}")

    print()



# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        sess = wizard()
        run_pipeline(sess)
    except KeyboardInterrupt:
        print(f"\n\n  {C.YELLOW}Cancelled.{C.RESET}\n")
        sys.exit(0)
