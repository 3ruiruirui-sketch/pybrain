"""
Brain Tumor Auto-Loader for 3D Slicer  —  crash-safe version
=============================================================
HOW TO RUN:
  1. Open 3D Slicer
  2. View → Python Interactor  (or Ctrl+3)
  3. Paste this ONE line and press Enter:
       exec(open('/Users/ssoares/Downloads/load_in_slicer.py').read())

If Slicer still crashes, run blocks individually — copy each section
marked  # ── BLOCK N ──  separately into the Python console.
"""

import os
import time

# ─────────────────────────────────────────────────────────────────────────
# ⚙️  PATHS
# ─────────────────────────────────────────────────────────────────────────
import json
from pathlib import Path

# Since this runs inside Slicer via exec(), __file__ might be missing.
# We look for PYBRAIN_SESSION or default to the most recent run.
_env_path = os.environ.get("PYBRAIN_SESSION")
if _env_path and os.path.exists(_env_path):
    with open(_env_path) as f:
        _sess = json.load(f)
else:
    # default to most recent results folder
    _proj = Path("~/Downloads/PY-BRAIN").expanduser()
    if not _proj.exists():
        _proj = Path("~/documents/PY-BRAIN").expanduser()
    _res = _proj / "results"
    _sess_files = sorted(_res.glob("*/session.json"))
    if _sess_files:
        with open(_sess_files[-1]) as f:
            _sess = json.load(f)
    else:
        print("ERROR: Could not find session.json in results/")
        _sess = {}

MRI_DIR = _sess.get("monai_dir", "")
EXTRA_DIR = _sess.get("extra_dir", "")
RESULTS_DIR = _sess.get("output_dir", "")


# ─────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────
def ui_update(label="", pause=0.3):
    slicer.app.processEvents()
    if pause > 0:
        time.sleep(pause)
    if label:
        print(f"  OK  {label}")


def safe_load_volume(path, name):
    if not os.path.exists(path):
        print(f"  --  Skip {name} (not found)")
        return None
    try:
        node = slicer.util.loadVolume(path)
        node.SetName(name)
        ui_update(f"Loaded {name}", pause=0.6)
        return node
    except Exception as e:
        print(f"  !!  {name} load error: {e}")
        return None


def safe_load_seg(path, name):
    if not os.path.exists(path):
        print(f"  --  Skip {name} (not found)")
        return None
    try:
        node = slicer.util.loadSegmentation(path)
        node.SetName(name)
        ui_update(f"Loaded {name}", pause=1.0)
        return node
    except Exception as e:
        print(f"  !!  {name} load error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────
print("=" * 55)
print("  Brain Tumor Loader  (crash-safe)")
print("=" * 55)

# ── BLOCK 1 — Clear scene ─────────────────────────────────────────────
print("\n[1/6] Clearing scene...")
try:
    slicer.mrmlScene.Clear(0)
    ui_update("Scene cleared", pause=1.5)
except Exception as e:
    print(f"  !!  {e}")

# ── BLOCK 2 — Load MRI volumes ────────────────────────────────────────
print("\n[2/6] Loading MRI volumes...")
volumes = {}
mri_list = [
    ("t1c", MRI_DIR + "/t1c_resampled.nii.gz", "T1c"),
    ("t1", MRI_DIR + "/t1.nii.gz", "T1"),
    ("flair", MRI_DIR + "/flair_resampled.nii.gz", "FLAIR"),
    ("t2", MRI_DIR + "/t2_resampled.nii.gz", "T2"),
]
for key, path, label in mri_list:
    node = safe_load_volume(path, label)
    if node:
        volumes[key] = node

# ── BLOCK 3 — Load CT ─────────────────────────────────────────────────
print("\n[3/6] Loading CT (if available)...")
ct_files = {
    "CT": "ct_brain_registered.nii.gz",
    "CT_Calc": "ct_calcification.nii.gz",
    "CT_Haem": "ct_haemorrhage.nii.gz",
}
for key, fname in ct_files.items():
    path = os.path.join(RESULTS_DIR, fname)
    node = safe_load_volume(path, key)
    if node:
        volumes[key.lower()] = node

# ── BLOCK 4 — Load segmentation ───────────────────────────────────────
print("\n[4/6] Loading segmentation...")
seg_merged = RESULTS_DIR + "/segmentation_ct_merged.nii.gz"
seg_brats = RESULTS_DIR + "/segmentation_full.nii.gz"
seg_path = seg_merged if os.path.exists(seg_merged) else seg_brats
seg_label = "BraTS+CT" if os.path.exists(seg_merged) else "BraTS"
print(f"  Using: {os.path.basename(seg_path)}")

seg_node = safe_load_seg(seg_path, seg_label)

if seg_node:
    try:
        seg = seg_node.GetSegmentation()
        cfg = {
            "1": ("Necrotic", [0.27, 0.53, 1.00]),
            "2": ("Edema", [0.27, 0.87, 0.27]),
            "3": ("Enhancing", [1.00, 0.27, 0.27]),
        }
        for i in range(seg.GetNumberOfSegments()):
            sid = seg.GetNthSegmentID(i)
            s = seg.GetSegment(sid)
            for lv, (nm, col) in cfg.items():
                if lv in s.GetName():
                    s.SetName(nm)
                    s.SetColor(col[0], col[1], col[2])
                    print(f"  Label {lv} -> {nm}")
                    break
        dn = seg_node.GetDisplayNode()
        if dn:
            dn.SetOpacity2DFill(0.50)
            dn.SetOpacity2DOutline(1.00)
            dn.SetVisibility(True)
        ui_update("Segments coloured", pause=0.5)
    except Exception as e:
        print(f"  !!  Colouring: {e}")

# ── BLOCK 5 — Layout ──────────────────────────────────────────────────
print("\n[5/6] Setting layout...")
try:
    lm = slicer.app.layoutManager()
    lm.setLayout(3)  # built-in Four-Up — safest option
    ui_update("Four-Up layout", pause=0.5)

    assignments = {"Red": volumes.get("t1c"), "Yellow": volumes.get("flair"), "Green": volumes.get("t2")}

    for panel, vol in assignments.items():
        if vol is None:
            continue
        try:
            sw = lm.sliceWidget(panel)
            if sw:
                sw.sliceLogic().GetSliceCompositeNode().SetBackgroundVolumeID(vol.GetID())
                sw.sliceLogic().FitSliceToAll()
                sw.mrmlSliceNode().SetLinkedControl(True)
                ui_update(f"{panel} = {vol.GetName()}", pause=0.2)
        except Exception as e:
            print(f"  !!  {panel}: {e}")
except Exception as e:
    print(f"  !!  Layout: {e}")

# ── BLOCK 6 — Jump to tumour ──────────────────────────────────────────
print("\n[6/6] Jumping to tumour centre...")
try:
    import numpy as np

    jumped = False
    if seg_node and volumes.get("t1c"):
        for lbl in ["3", "2", "1"]:
            try:
                arr = slicer.util.arrayFromSegmentBinaryLabelmap(seg_node, lbl, volumes["t1c"])
                if arr is not None and arr.max() > 0:
                    c = np.argwhere(arr > 0).mean(axis=0).astype(int)
                    m = vtk.vtkMatrix4x4()
                    volumes["t1c"].GetIJKToRASMatrix(m)
                    r = m.MultiplyPoint([float(c[2]), float(c[1]), float(c[0]), 1.0])
                    slicer.modules.markups.logic().JumpSlicesToLocation(r[0], r[1], r[2], True)
                    ui_update(f"Jumped to label {lbl} centre", pause=0.3)
                    jumped = True
                    break
            except:
                continue
    if not jumped:
        print("  --  Could not jump — scroll manually")
except Exception as e:
    print(f"  !!  Jump: {e}")

# ── Done ──────────────────────────────────────────────────────────────
ui_update(pause=0.5)
print("\n" + "=" * 55)
print("  Scene loaded!")
print("=" * 55)
print("""
  PANELS:
    Red    (Axial)    = T1c   (enhancing tumour)
    Yellow (Sagittal) = FLAIR (oedema)
    Green  (Coronal)  = T2    (full extent)

  TUMOUR COLOURS:
    Blue  = Necrotic core
    Green = Peritumoral oedema
    Red   = Enhancing tumour

  ENABLE 3D RENDER (do manually to avoid crash):
    Left panel -> click eye icon next to segmentation

  VOLUME RENDERING:
    Modules -> Volume Rendering -> T1c -> Enable

  CREATE GROUND TRUTH:
    Modules -> Segment Editor -> Paint/Draw
    File -> Save Data -> ground_truth.nii.gz
    Folder: ~/documents/PY-BRAIN/results/
""")
