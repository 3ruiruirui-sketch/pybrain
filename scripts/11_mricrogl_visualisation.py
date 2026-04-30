#!/usr/bin/env python3
"""
Stage 11 — MRIcroGL Advanced 3D Visualisation
================================================
Integrates MRIcroGL (https://github.com/rordenlab/MRIcroGL)
for GPU-accelerated volume rendering of all PY-BRAIN data.

MRIcroGL uses GLSL/Metal shaders on Apple Silicon for
real-time ray-casting volume rendering — far superior to
the marching-cubes mesh in Stage 10.

What this script does:
  1. Installs MRIcroGL if not present
  2. Generates a Python script in MRIcroGL's scripting language
  3. Launches MRIcroGL with all patient volumes pre-loaded
  4. Renders and saves high-quality PNG images automatically:
       - Volume rendering (T1c + tumour overlay)
       - MIP (maximum intensity projection)
       - Mosaic (all slices grid)
       - Glass brain (transparent rendering)
       - CT + MRI fusion
       - All 4 MRI modalities side by side
  5. Saves renders to results folder

MRIcroGL scripting uses 'import gl' — it runs INSIDE MRIcroGL,
not in the regular Python interpreter.

Run:  python3 scripts/11_mricrogl_visualisation.py
"""

import sys
import shutil
import subprocess
from pathlib import Path

# ── Session ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.session_loader import get_session, get_paths  # type: ignore

sess = get_session()
paths = get_paths(sess)
MONAI = paths["monai_dir"]
EXTRA = paths["extra_dir"]
OUT = paths["output_dir"]
PATIENT = sess.get("patient", {})
PNAME = PATIENT.get("name", "Patient").replace(" ", "_")[:20]


def banner(t):
    print("\n" + "═" * 65)
    print(f"  {t}")
    print("═" * 65)


# ═════════════════════════════════════════════════════════════════════════
# STEP 1 — Find or install MRIcroGL
# ═════════════════════════════════════════════════════════════════════════
banner("STEP 1 — FINDING MRIcroGL")

MRICROGL_APP = Path("/Applications/MRIcroGL.app")
MRICROGL_BIN = MRICROGL_APP / "Contents" / "MacOS" / "MRIcroGL"
MRICROGL_ALT = Path.home() / "Applications" / "MRIcroGL.app" / "Contents" / "MacOS" / "MRIcroGL"

# Check common locations
mricrogl_exe = None
for candidate in [MRICROGL_BIN, MRICROGL_ALT, Path("/usr/local/bin/mricrogl"), Path("/opt/homebrew/bin/mricrogl")]:
    if candidate.exists():
        mricrogl_exe = candidate
        break

# Also check if 'mricrogl' is on PATH
bin_path = shutil.which("mricrogl")
if mricrogl_exe is None and bin_path:
    mricrogl_exe = Path(bin_path)

if mricrogl_exe:
    print(f"  ✅ MRIcroGL found: {mricrogl_exe}")
else:
    print("  ⚠️  MRIcroGL not found.")
    print(
        """
  Install MRIcroGL for Mac (Apple Silicon):
  ─────────────────────────────────────────
  Option A — Download directly (recommended):
    1. Go to: https://github.com/rordenlab/MRIcroGL/releases
    2. Download: MRIcroGL_macOS.zip  (choose aarch64 for Apple Silicon M-series)
    3. Unzip and drag MRIcroGL.app to /Applications/
    4. Run this script again

  Option B — Homebrew:
    brew install --cask mricrogl

  Option C — Run manually (without this script):
    Open MRIcroGL → drag your NIfTI files from Finder
    Files are in: {out}
""".format(out=OUT)
    )

    # Create a drag-and-drop instructions file
    instructions = OUT / "MRICROGL_INSTRUCTIONS.txt"
    t1c_path = MONAI / "t1c_resampled.nii.gz"
    seg_path = OUT / "segmentation_full.nii.gz"
    ct_path = EXTRA / "ct_brain_resampled.nii.gz"

    with open(instructions, "w") as f:
        f.write("MRIcroGL Manual Loading Instructions\n")
        f.write(f"Patient: {PATIENT.get('name', '?')}\n")
        f.write("=" * 50 + "\n\n")
        f.write("1. Open MRIcroGL\n\n")
        f.write("2. Drag these files into MRIcroGL (in order):\n\n")
        f.write("   BACKGROUND (main volume):\n")
        f.write(f"   {t1c_path}\n\n")
        f.write("   OVERLAY 1 (tumour segmentation):\n")
        f.write(f"   {seg_path}\n\n")
        if ct_path.exists():
            f.write("   OVERLAY 2 (CT — registered to MRI):\n")
            f.write(f"   {ct_path}\n\n")
        f.write("3. View → Rendering (or press R)\n")
        f.write("   → You will see 3D volume rendering\n\n")
        f.write("4. In the rendering window:\n")
        f.write("   → Drag to rotate\n")
        f.write("   → Scroll to zoom\n")
        f.write("   → Right-click for rendering options\n\n")
        f.write("5. For the tumour overlay:\n")
        f.write("   → Layers panel → set colour map to 'actc'\n")
        f.write("   → Adjust opacity slider\n\n")
        f.write("6. File → Save Image to save screenshots\n\n")
        f.write("All NIfTI files are in:\n")
        f.write(f"  {OUT}\n")
        f.write(f"  {MONAI}\n")
        f.write(f"  {EXTRA}\n")

    print(f"  📄 Manual instructions saved → {instructions}")
    print("\n  Install MRIcroGL then run this script again for auto-rendering.")
    sys.exit(0)


# ═════════════════════════════════════════════════════════════════════════
# STEP 2 — Collect all available NIfTI files
# ═════════════════════════════════════════════════════════════════════════
banner("STEP 2 — COLLECTING NIfTI FILES")


def find_file(*candidates):
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    return None


t1c = find_file(MONAI / "t1c_resampled.nii.gz", MONAI / "t1c.nii.gz")
t1 = find_file(MONAI / "t1_resampled.nii.gz", MONAI / "t1.nii.gz")
t2 = find_file(MONAI / "t2_resampled.nii.gz", MONAI / "t2.nii.gz")
flair = find_file(MONAI / "flair_resampled.nii.gz", MONAI / "flair.nii.gz")
# Standardized segmentation source for cross-stage fallback
# Ensure we prioritize the current session output
SEG_DIR = Path(str(paths.get("seg_dir", OUT)))

seg = find_file(
    SEG_DIR / "segmentation_full.nii.gz",
    SEG_DIR / "segmentation_ensemble.nii.gz",
    SEG_DIR / "segmentation_ct_merged.nii.gz",
)
ct = find_file(EXTRA / "ct_brain_resampled.nii.gz", EXTRA / "ct_brain.nii.gz", EXTRA / "ct_brain_registered.nii.gz")
calc = find_file(EXTRA / "ct_calcification.nii.gz")
prob = find_file(
    SEG_DIR / "ensemble_probability.nii.gz", SEG_DIR / "prob_wt.nii.gz", SEG_DIR / "tumor_probability.nii.gz"
)

for name, path in [
    ("T1c", t1c),
    ("T1", t1),
    ("T2", t2),
    ("FLAIR", flair),
    ("Segmentation", seg),
    ("CT", ct),
    ("Calcification", calc),
    ("Probability", prob),
]:
    status = f"✅ {path}" if path else "⚠️  not found"
    print(f"  {name:15s}: {status}")

if t1c is None and t1 is None:
    print("❌  No MRI volumes found — run Stage 1 first")
    sys.exit(1)

bg_vol = t1c or t1  # use T1c as background if available


# ═════════════════════════════════════════════════════════════════════════
# STEP 3 — Generate MRIcroGL Python scripts
# ═════════════════════════════════════════════════════════════════════════
banner("STEP 3 — GENERATING MRIcroGL SCRIPTS")

SCRIPT_DIR = OUT / "mricrogl_scripts"
RENDER_DIR = OUT / "mricrogl_renders"
SCRIPT_DIR.mkdir(exist_ok=True)
RENDER_DIR.mkdir(exist_ok=True)


def p(path):
    """Return path as forward-slash string for MRIcroGL."""
    return str(path).replace("\\", "/")


# ── Script 1: Volume rendering + tumour overlay ───────────────────────────
script1 = SCRIPT_DIR / "render_tumour.py"
with open(script1, "w") as f:
    f.write(f'''import gl
import time

# PY-BRAIN — Tumour Volume Rendering
# Patient: {PATIENT.get("name", "?")}
# Auto-generated by 11_mricrogl_visualisation.py

gl.resetdefaults()
gl.windowposition(0, 0, 900, 700)
gl.toolformvisible(0)

# Load T1c as background volume
gl.loadimage("{p(bg_vol)}")

# Rendering settings — volume ray casting
gl.shadername("phong")
gl.shaderquality1to10(8)
gl.shaderadjust("ambient", 0.3)
gl.shaderadjust("diffuse", 0.7)
gl.shaderlightazimuthelevation(0, 60)

# Set display range for T1c (brain tissue window)
gl.minmax(0, 200, 1200)

# Load tumour segmentation as overlay
''')
    if seg:
        f.write(f'''gl.overlayload("{p(seg)}")
gl.colorname(1, "actc")       # actc = activation-style warm colours
gl.minmax(1, 0.5, 3.5)        # show labels 1,2,3
gl.opacity(1, 80)              # 80% opacity

''')
    if ct:
        f.write(f'''# Load CT as second overlay (shows calcifications in bone colour)
gl.overlayload("{p(ct)}")
gl.colorname(2, "gray")
gl.minmax(2, 130, 1000)        # show calcification range only
gl.opacity(2, 40)

''')
    f.write(f'''# View from left side — shows left fronto-parietal tumour
gl.viewsagittal(1)
gl.shaderquality1to10(9)
time.sleep(0.5)
gl.savebmp("{p(RENDER_DIR)}/{PNAME}_render_sagittal.png")

# View from front
gl.viewcoronal(0)
time.sleep(0.5)
gl.savebmp("{p(RENDER_DIR)}/{PNAME}_render_coronal.png")

# View from top
gl.viewaxial(1)
time.sleep(0.5)
gl.savebmp("{p(RENDER_DIR)}/{PNAME}_render_axial.png")

# 3/4 oblique view — most dramatic for report
gl.viewsagittal(1)
gl.azimuthelevation(30, 20)
time.sleep(0.5)
gl.savebmp("{p(RENDER_DIR)}/{PNAME}_render_oblique.png")

print("Volume rendering complete — 4 views saved")
''')

# ── Script 2: MIP (Maximum Intensity Projection) ──────────────────────────
script2 = SCRIPT_DIR / "render_mip.py"
with open(script2, "w") as f:
    f.write(f'''import gl
import time

# MIP — Maximum Intensity Projection
gl.resetdefaults()
gl.windowposition(0, 0, 900, 700)
gl.toolformvisible(0)
gl.loadimage("{p(bg_vol)}")
gl.shadername("mip")           # MIP shader
gl.shaderquality1to10(9)
gl.minmax(0, 300, 1500)

''')
    if seg:
        f.write(f'''gl.overlayload("{p(seg)}")
gl.colorname(1, "actc")
gl.minmax(1, 0.5, 3.5)
gl.opacity(1, 90)
''')
    f.write(f'''
for az in [0, 45, 90, 135, 180, 225, 270, 315]:
    gl.azimuthelevation(az, 15)
    time.sleep(0.3)
    gl.savebmp("{p(RENDER_DIR)}/{PNAME}_mip_az{{:03d}}.png".format(az))

print("MIP rotation complete — 8 views saved")
''')

# ── Script 3: Mosaic (slice grid) ─────────────────────────────────────────
script3 = SCRIPT_DIR / "render_mosaic.py"
with open(script3, "w") as f:
    f.write(f'''import gl

# Mosaic — all slices grid
gl.resetdefaults()
gl.windowposition(0, 0, 1200, 900)
gl.toolformvisible(0)
gl.loadimage("{p(bg_vol)}")

''')
    if seg:
        f.write(f'''gl.overlayload("{p(seg)}")
gl.colorname(1, "actc")
gl.minmax(1, 0.5, 3.5)
gl.opacity(1, 70)
''')
    f.write(f'''
# Axial mosaic — all slices
gl.mosaic("A L 6 6")           # Axial, Left=brain left, 6x6 grid
gl.savebmp("{p(RENDER_DIR)}/{PNAME}_mosaic_axial.png")

# Coronal mosaic
gl.mosaic("C L 6 6")
gl.savebmp("{p(RENDER_DIR)}/{PNAME}_mosaic_coronal.png")

# Sagittal mosaic
gl.mosaic("S L 6 6")
gl.savebmp("{p(RENDER_DIR)}/{PNAME}_mosaic_sagittal.png")

print("Mosaic views saved")
''')

# ── Script 4: Glass brain ─────────────────────────────────────────────────
script4 = SCRIPT_DIR / "render_glass.py"
with open(script4, "w") as f:
    f.write(f'''import gl
import time

# Glass brain — transparent rendering showing internal structures
gl.resetdefaults()
gl.windowposition(0, 0, 900, 700)
gl.toolformvisible(0)
gl.loadimage("{p(bg_vol)}")
gl.shadername("glass")         # glass shader — see through
gl.shaderquality1to10(8)
gl.opacity(0, 15)              # very transparent background
gl.minmax(0, 200, 1500)

''')
    if seg:
        f.write(f'''gl.overlayload("{p(seg)}")
gl.colorname(1, "actc")
gl.minmax(1, 0.5, 3.5)
gl.opacity(1, 95)              # tumour fully opaque inside glass brain
''')
    f.write(f'''
gl.viewsagittal(1)
time.sleep(0.5)
gl.savebmp("{p(RENDER_DIR)}/{PNAME}_glass_sagittal.png")

gl.viewcoronal(0)
time.sleep(0.5)
gl.savebmp("{p(RENDER_DIR)}/{PNAME}_glass_coronal.png")

gl.azimuthelevation(30, 20)
time.sleep(0.5)
gl.savebmp("{p(RENDER_DIR)}/{PNAME}_glass_oblique.png")

print("Glass brain views saved")
''')

# ── Script 5: CT + MRI fusion ─────────────────────────────────────────────
script5 = SCRIPT_DIR / "render_ct_mri_fusion.py"
if ct:
    with open(script5, "w") as f:
        f.write(f'''import gl
import time

# CT + MRI fusion — shows calcifications in context of tumour
gl.resetdefaults()
gl.windowposition(0, 0, 900, 700)
gl.toolformvisible(0)

# T1c as background
gl.loadimage("{p(bg_vol)}")
gl.minmax(0, 0, 1200)

# CT overlay — bone window to show calcifications
gl.overlayload("{p(ct)}")
gl.colorname(1, "nih")         # NIH colour scale
gl.minmax(1, 80, 500)          # brain + calcification range
gl.opacity(1, 50)

''')
        if calc:
            f.write(f'''# Calcification mask overlay
gl.overlayload("{p(calc)}")
gl.colorname(2, "gold")        # gold for calcifications
gl.minmax(2, 0.5, 1.5)
gl.opacity(2, 85)

''')
        if seg:
            f.write(f'''# Tumour segmentation
gl.overlayload("{p(seg)}")
gl.colorname(3, "actc")
gl.minmax(3, 0.5, 3.5)
gl.opacity(3, 60)

''')
        f.write(f'''gl.viewsagittal(1)
time.sleep(0.5)
gl.savebmp("{p(RENDER_DIR)}/{PNAME}_ct_mri_fusion_sagittal.png")

gl.azimuthelevation(30, 20)
time.sleep(0.5)
gl.savebmp("{p(RENDER_DIR)}/{PNAME}_ct_mri_fusion_oblique.png")

print("CT-MRI fusion renders saved")
''')

# ── Script 6: All 4 modalities mosaic ────────────────────────────────────
script6 = SCRIPT_DIR / "render_all_modalities.py"
with open(script6, "w") as f:
    modality_paths = [(n, p(v)) for n, v in [("T1c", t1c), ("T1", t1), ("T2", t2), ("FLAIR", flair)] if v is not None]
    f.write("""import gl

# All 4 MRI modalities — axial mosaic comparison
gl.resetdefaults()
gl.windowposition(0, 0, 1400, 400)
gl.toolformvisible(0)

""")
    for i, (mod_name, mod_path) in enumerate(modality_paths):
        f.write(f'''# {mod_name}
gl.loadimage("{mod_path}")
gl.mosaic("A L 1 8")
gl.savebmp("{p(RENDER_DIR)}/{PNAME}_mosaic_{mod_name}.png")

''')
    f.write('print("All modality mosaics saved")\n')

# ── Master script — runs all the above ───────────────────────────────────
script_master = SCRIPT_DIR / "render_all.py"
with open(script_master, "w") as f:
    f.write(f'''import gl
import os

# PY-BRAIN — Master render script
# Runs all visualisation scripts in sequence
# Patient: {PATIENT.get("name", "?")}

scripts = [
    "{p(script1)}",
    "{p(script2)}",
    "{p(script3)}",
    "{p(script4)}",
    "{p(script5) if ct else ""}",
    "{p(script6)}",
]

for s in scripts:
    if s and os.path.exists(s):
        print(f"Running: {{s}}")
        exec(open(s).read())

print("All PY-BRAIN renders complete!")
print("Saved to: {p(RENDER_DIR)}")
''')

for script, name in [
    (script1, "Volume rendering"),
    (script2, "MIP rotation"),
    (script3, "Mosaic grids"),
    (script4, "Glass brain"),
    (script5 if ct else None, "CT-MRI fusion"),
    (script6, "All modalities"),
    (script_master, "Master script"),
]:
    if script:
        print(f"  ✅ Generated: {script.name}  — {name}")


# ═════════════════════════════════════════════════════════════════════════
# STEP 4 — Launch MRIcroGL
# ═════════════════════════════════════════════════════════════════════════
banner("STEP 4 — LAUNCHING MRIcroGL")

print(f"  MRIcroGL   : {mricrogl_exe}")
print(f"  Script     : {script_master}")
print(f"  Output dir : {RENDER_DIR}")
print()


# On macOS, MRIcroGL.app needs special launch method
def launch_mricrogl(script_path: Path):
    """Launch MRIcroGL with a Python script on macOS."""
    if str(mricrogl_exe).endswith("MacOS/MRIcroGL"):
        # Direct binary call
        cmd = [str(mricrogl_exe), str(script_path)]
    else:
        cmd = [str(mricrogl_exe), str(script_path)]

    print(f"  Running: {' '.join(cmd)}\n")
    try:
        result = subprocess.run(cmd, timeout=300, capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✅ MRIcroGL completed successfully")
            return True
        else:
            print(f"  ⚠️  MRIcroGL exit code: {result.returncode}")
            if result.stderr:
                err_msg: str = str(result.stderr)
                print(f"  stderr: {err_msg[:500]}")  # type: ignore
            return False
    except subprocess.TimeoutExpired:
        print("  ⚠️  Timeout — MRIcroGL took > 5 min")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


success = launch_mricrogl(script_master)


# ═════════════════════════════════════════════════════════════════════════
# STEP 5 — Report what was generated
# ═════════════════════════════════════════════════════════════════════════
banner("STEP 5 — RESULTS")

renders = list(RENDER_DIR.glob("*.png"))
if renders:
    print(f"\n  {len(renders)} renders saved to: {RENDER_DIR}\n")
    for r in sorted(renders):
        size = r.stat().st_size / 1024
        print(f"  ✅ {r.name:<55} {size:.0f} KB")
else:
    print(f"\n  No renders found in {RENDER_DIR}")
    print("""
  Possible reasons:
    1. MRIcroGL opened but rendering took too long
       → Open MRIcroGL manually and run the script:
         Scripting → Open → select render_all.py

    2. OpenGL/Metal issue on this Mac
       → Try running MRIcroGL manually first to confirm it works

    3. Script error in MRIcroGL Python bridge
       → Check MRIcroGL scripting console for errors
""")

print(f"""
  MRIcroGL scripts saved to:
    {SCRIPT_DIR}

  To run manually inside MRIcroGL:
    Scripting → Open → {script_master.name}
    Scripting → Run

  Or from Terminal:
    '{mricrogl_exe}' '{script_master}'
""")

print("═" * 65)
print("  ✅  Stage 11 done")
print("═" * 65)
