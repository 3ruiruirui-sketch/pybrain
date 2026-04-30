#!/usr/bin/env python3
"""
scripts/0_preflight_check.py
===========================
Full preflight validation script for the brain tumour analysis pipeline.
Checks Python dependencies, input files, geometric compatibility,
content quality, and clinical consistency.
"""

import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import nibabel as nib
import numpy as np

# ── Project imports ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.session_loader import get_session, get_paths, get_patient
except ImportError:
    from session_loader import get_session, get_paths, get_patient


# ── ANSI Colours ──────────────────────────────────────────────────────────
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"


def banner(title: str):
    """Prints a colourful section banner."""
    rule = f"{C.BOLD}{C.CYAN}{'═' * 70}{C.RESET}"
    print(f"\n{rule}")
    print(f"{C.BOLD}{C.CYAN}  {title}{C.RESET}")
    print(rule)


def status(category: str, check_name: str, level: str, msg: str):
    """Prints a status line and returns the status char for summary."""
    icons = {"OK": f"{C.GREEN}✅{C.RESET}", "WARN": f"{C.YELLOW}⚠️ {C.RESET}", "ERROR": f"{C.RED}❌{C.RESET}"}
    cat_str = f"[{category:12s}]"
    print(f"  {icons.get(level, ' ')} {C.BOLD}{cat_str}{C.RESET} {check_name:30s} ... {msg}")
    return level


# ── Main Class ──────────────────────────────────────────────────────────
class Preflight:
    def __init__(self):
        self.sess = get_session()
        self.paths = get_paths(self.sess)
        self.vols: Dict[str, Any] = {}
        self.results = []
        self.summary = {"total": 0, "OK": 0, "WARN": 0, "ERROR": 0}

        self.output_dir = self.paths["output_dir"]
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log(self, category: str, name: str, level: str, msg: str):
        self.results.append(
            {
                "category": category,
                "name": name,
                "level": level,
                "message": msg,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self.summary["total"] += 1
        self.summary[level] += 1
        return status(category, name, level, msg)

    # ─────────────────────────────────────────────────────────────────
    # LAYER 1 — Python dependencies + external tools
    # ─────────────────────────────────────────────────────────────────
    def check_layer1(self):
        banner("LAYER 1 — DEPENDENCIES & EXTERNAL TOOLS")

        req_pkgs = ["numpy", "nibabel", "SimpleITK", "scipy", "torch", "monai", "skimage", "pydicom", "plotly", "yaml"]
        alt_names = {"skimage": "scikit-image", "yaml": "PyYAML"}

        import importlib.util
        import importlib.metadata

        for pkg in req_pkgs:
            spec = importlib.util.find_spec(pkg)
            if spec is not None:
                try:
                    import importlib.metadata

                    ver = importlib.metadata.version(alt_names.get(pkg, pkg))
                    self.log("DEPS", pkg, "OK", f"Found v{ver}")
                except Exception:
                    self.log("DEPS", pkg, "OK", "Found (version unknown)")
            else:
                self.log("DEPS", pkg, "ERROR", f"Missing package: {pkg}")

        # pybrain Package Check
        pybrain_spec = importlib.util.find_spec("pybrain")
        if pybrain_spec:
            self.log("PACKAGE", "pybrain", "OK", "Modular package found")

            # Check for config files
            cfg_dir = PROJECT_ROOT / "pybrain" / "config"
            if (cfg_dir / "defaults.yaml").exists() and (cfg_dir / "hardware_profiles.yaml").exists():
                self.log("PACKAGE", "pybrain.config", "OK", "YAML configurations found")
            else:
                self.log("PACKAGE", "pybrain.config", "ERROR", "Missing YAML configurations in pybrain/config/")
        else:
            self.log("PACKAGE", "pybrain", "ERROR", "Modular package 'pybrain' not found in path")

        # External tools
        d2n = shutil.which("dcm2niix")
        if d2n:
            self.log("TOOLS", "dcm2niix", "OK", f"Found at {d2n}")
        else:
            self.log("TOOLS", "dcm2niix", "ERROR", "Missing dcm2niix (required for Stage 1/2)")

        fsl = shutil.which("fslmaths")
        if fsl:
            self.log("TOOLS", "fslmaths", "OK", "Found")
        else:
            self.log("TOOLS", "fslmaths", "WARN", "Missing fslmaths (optional)")

    # ─────────────────────────────────────────────────────────────────
    # LAYER 2 — Input files existence + readability
    # ─────────────────────────────────────────────────────────────────
    def check_layer2(self):
        banner("LAYER 2 — INPUT FILES EXISTENCE & READABILITY")

        monai_dir = self.paths["monai_dir"]
        extra_dir = self.paths["extra_dir"]

        # BRAINMODALITIES: T1, T1c, T2, FLAIR
        # Standardized volumes generated by Stage 1b
        required = {
            "T1": monai_dir / "t1_resampled.nii.gz",
            "T1c": monai_dir / "t1c_resampled.nii.gz",
            "T2": monai_dir / "t2_resampled.nii.gz",
            "FLAIR": monai_dir / "flair_resampled.nii.gz",
        }

        # Determine the most likely 'Standard Shape' reference
        standard_ref = None
        for label in ["T1c", "T1", "T2", "FLAIR"]:
            p = monai_dir / f"{label.lower()}_resampled.nii.gz"
            if p.exists():
                standard_ref = p
                break

        # If no resampled files exist yet, use raw T1 as the baseline
        if not standard_ref:
            standard_ref = monai_dir / "t1.nii.gz"

        # Update required["T1"] for SNR/Coverage checks
        if not required["T1"].exists() and standard_ref.name == "t1.nii.gz":
            required["T1"] = monai_dir / "t1.nii.gz"
        elif not required["T1"].exists() and standard_ref:
            required["T1"] = standard_ref  # Use whatever standardized ref we have

        # Set the reference for all subsequent shape checks
        self.ref_path = standard_ref

        optional = {
            "ADC": extra_dir / "adc_resampled.nii.gz",
            "DWI": extra_dir / "dwi_resampled.nii.gz",
            "T2*": extra_dir / "t2star_resampled.nii.gz",
        }

        is_stage_1 = self.sess.get("stages", {}).get("stage_1_dicom", False)

        # Check required
        self.vols = {}
        for label, path in required.items():
            if not path.exists():
                if is_stage_1:
                    self.log("INPUT", f"{label} NIfTI", "WARN", "Missing, but Stage 1 will generate it.")
                else:
                    self.log("INPUT", f"{label} NIfTI", "ERROR", f"Missing: {path.name}")
            else:
                try:
                    img = nib.load(str(path))
                    self.vols[label] = img
                    size_mb = path.stat().st_size / (1024 * 1024)
                    msg = f"{img.shape} | {img.get_data_dtype()} | {size_mb:.1f} MB"
                    self.log("INPUT", f"{label} NIfTI", "OK", msg)
                except Exception as e:
                    self.log("INPUT", f"{label} NIfTI", "ERROR", f"Unreadable: {e}")

        # Check optional
        for label, path in optional.items():
            if not path.exists():
                self.log("INPUT", f"{label} NIfTI", "WARN", f"Missing (optional): {path.name}")
            else:
                try:
                    img = nib.load(str(path))
                    self.vols[label] = img
                    self.log("INPUT", f"{label} NIfTI", "OK", f"Found {img.shape}")
                except Exception:
                    self.log("INPUT", f"{label} NIfTI", "WARN", "Unreadable")

        # Check Stage Output (Segmentation)
        seg_names = ["segmentation_full.nii.gz", "segmentation_ensemble.nii.gz"]
        found_seg = None
        for name in seg_names:
            p = self.output_dir / name
            if p.exists():
                found_seg = p
                break

        if found_seg is not None:
            try:
                self.vols["SEG"] = nib.load(str(found_seg))
                self.log("OUTPUT", "Segmentation", "OK", f"Found {found_seg.name}")
            except Exception:
                self.log("OUTPUT", "Segmentation", "ERROR", "Unreadable")
        else:
            is_stage_3 = self.sess.get("stages", {}).get("stage_3_segment", False)
            if is_stage_3:
                self.log("OUTPUT", "Segmentation", "WARN", "Missing, but Stage 3 will generate it")
            else:
                self.log("OUTPUT", "Segmentation", "WARN", "Missing AI segmentation")

    # ─────────────────────────────────────────────────────────────────
    # LAYER 3 — Geometric compatibility
    # ─────────────────────────────────────────────────────────────────
    def check_layer3(self):
        banner("LAYER 3 — GEOMETRIC COMPATIBILITY")

        is_stage_1 = self.sess.get("stages", {}).get("stage_1_dicom", False)

        if "T1" not in self.vols:
            if is_stage_1:
                self.log("GEOM", "Reference T1", "WARN", "Cannot check geometry (Waiting on Stage 1)")
            else:
                self.log("GEOM", "Reference T1", "ERROR", "Cannot check geometry without T1")
            return

        ref_img = self.vols["T1"]
        ref_shape = ref_img.shape
        ref_affine = ref_img.affine
        ref_pixdim = ref_img.header.get_zooms()[:3]

        # Shape & Affine match
        for label, img in self.vols.items():
            if label == "T1":
                continue

            # Shape match logic
            if img.shape != ref_shape:
                is_stage_1b = self.sess.get("stages", {}).get("stage_1b_prep", False)
                # If we are going to run Stage 1b (Refinement), shape mismatch is a WARN, not an ERROR
                # because Stage 1b will fix it.
                level = "ERROR" if (label in ["T1", "T1c", "T2", "FLAIR", "SEG"] and not is_stage_1b) else "WARN"
                self.log("GEOM", f"{label} shape", level, f"Mismatch: {img.shape} vs Ref {ref_shape}")
            else:
                self.log("GEOM", f"{label} shape", "OK", "Match")

            # Affine
            diff = np.abs(img.affine - ref_affine).max()
            if diff > 0.5:
                self.log("GEOM", f"{label} affine", "WARN", f"Shift > 0.5mm (max diff: {diff:.3f}mm)")
            else:
                self.log("GEOM", f"{label} affine", "OK", "Match (< 0.5mm)")

        # Isotropic check
        if not np.all(np.abs(np.diff(ref_pixdim)) < 0.1):
            self.log("GEOM", "Isotropic", "WARN", f"Resolution non-isotropic: {ref_pixdim}")
        else:
            self.log("GEOM", "Isotropic", "OK", f"Resolution: {ref_pixdim}")

    # ─────────────────────────────────────────────────────────────────
    # LAYER 4 — Content quality
    # ─────────────────────────────────────────────────────────────────
    def check_layer4(self):
        banner("LAYER 4 — CONTENT QUALITY")

        for label, img in self.vols.items():
            if label == "SEG":
                continue

            data = img.get_fdata()
            # Empty
            if data.max() <= 0:
                self.log("QUALITY", f"{label} empty", "ERROR", "Volume is zero/negative")

            # NaN / Inf
            if np.isnan(data).any() or np.isinf(data).any():
                self.log("QUALITY", f"{label} values", "ERROR", "Found NaN or Inf")

            # SNR
            signal = data[data > (0.1 * data.max())]
            if len(signal) > 0:
                s_mean = signal.mean()
                # Simple background noise estimate (bottom 5% intensity)
                noise_thresh = np.percentile(data, 5)
                bg = data[data <= noise_thresh]
                s_noise = bg.std() if len(bg) > 0 else 1.0
                snr = s_mean / max(s_noise, 0.001)

                if snr < 5.0:
                    self.log("QUALITY", f"{label} SNR", "WARN", f"Low SNR: {snr:.2f}")
                else:
                    self.log("QUALITY", f"{label} SNR", "OK", f"SNR {snr:.2f}")

            # Nonzero %
            pct = (data > 0).sum() / data.size * 100
            if pct < 10:
                self.log("QUALITY", f"{label} coverage", "WARN", f"Low coverage ({pct:.1f}%) — over-stripped?")
            elif pct > 60:
                self.log("QUALITY", f"{label} coverage", "WARN", f"High coverage ({pct:.1f}%) — no skull-strip?")
            else:
                self.log("QUALITY", f"{label} coverage", "OK", f"{pct:.1f}%")

        # Segmentation check
        if "SEG" in self.vols:
            seg_data = self.vols["SEG"].get_fdata().astype(np.uint8)
            unique_labels = np.unique(seg_data)
            # Two valid label conventions:
            #   Pipeline output:  {0,1,2,3}  — ET=3 (internal convention)
            #   BraTS 2021 GT:    {0,1,2,4}  — ET=4 (label 3 absent in training data)
            valid_pipeline = all(l in [0, 1, 2, 3] for l in unique_labels)
            valid_brats_gt = all(l in [0, 1, 2, 4] for l in unique_labels)
            if valid_pipeline:
                self.log("QUALITY", "SEG labels", "OK", "Labels in {0,1,2,3} (pipeline output)")
            elif valid_brats_gt:
                self.log("QUALITY", "SEG labels", "OK", "Labels in {0,1,2,4} (BraTS 2021 GT)")
            else:
                self.log("QUALITY", "SEG labels", "ERROR", f"Invalid labels found: {unique_labels}")

            # Volume
            pixdim = self.vols["SEG"].header.get_zooms()[:3]
            vox_cc = np.prod(pixdim) / 1000.0
            vol_cc = (seg_data > 0).sum() * vox_cc
            if vol_cc < 1.0:
                self.log("QUALITY", "SEG volume", "WARN", f"Suspiciously small: {vol_cc:.2f} cc")
            elif vol_cc > 300.0:
                self.log("QUALITY", "SEG volume", "WARN", f"Unrealistically large: {vol_cc:.2f} cc")
            else:
                self.log("QUALITY", "SEG volume", "OK", f"{vol_cc:.2f} cc")

    # ─────────────────────────────────────────────────────────────────
    # LAYER 5 — Clinical consistency
    # ─────────────────────────────────────────────────────────────────
    def check_layer5(self):
        banner("LAYER 5 — CLINICAL CONSISTENCY")

        # Patient info
        patient = get_patient(self.sess)
        if not patient.get("name") or not patient.get("exam_date"):
            self.log("CLINICAL", "Metadata", "WARN", "Missing patient name or exam date")
        else:
            self.log("CLINICAL", "Metadata", "OK", f"Patient: {patient.get('name')}")

        # Enhancement check
        if "T1" in self.vols and "T1c" in self.vols and "SEG" in self.vols:
            seg_data = self.vols["SEG"].get_fdata().astype(np.uint8)
            enh_mask = (seg_data == 4) | (seg_data == 3)  # BraTS2021 GT=4, pipeline output=3
            if np.sum(enh_mask) > 50:
                t1_data = self.vols["T1"].get_fdata()
                t1c_data = self.vols["T1c"].get_fdata()

                # Verify shape matching before indexing
                if t1_data.shape != enh_mask.shape or t1c_data.shape != enh_mask.shape:
                    self.log("CLINICAL", "Enhancement", "WARN", "Shape mismatch (SEG vs T1/T1c) — skipping check")
                else:
                    # Normalise for ratio (simple mean ratio of shared pixels)
                    m_t1 = t1_data[enh_mask].mean()
                    m_t1c = t1c_data[enh_mask].mean()
                    ratio = m_t1c / (m_t1 + 1e-8)

                    if ratio > 1.1:
                        self.log("CLINICAL", "Enhancement", "OK", f"Confirmed (Ratio {ratio:.2f})")
                    else:
                        self.log("CLINICAL", "Enhancement", "WARN", f"Weak/Absent (Ratio {ratio:.2f})")
            else:
                self.log("CLINICAL", "Enhancement", "OK", "N/A (No enhancing label)")

        # ADC check
        if "ADC" in self.vols and "SEG" in self.vols:
            seg_img = self.vols["SEG"]
            adc_img = self.vols["ADC"]
            if seg_img.shape == adc_img.shape:
                seg_data = seg_img.get_fdata().astype(np.uint8)
                tumor_mask = seg_data > 0
                if tumor_mask.any():
                    adc_data = adc_img.get_fdata()
                    m_adc = adc_data[tumor_mask].mean()
                    if m_adc < 800:
                        self.log("CLINICAL", "ADC Tumour", "OK", f"Mean {m_adc:.0f} -> High cellularity")
                    elif m_adc > 1200:
                        self.log("CLINICAL", "ADC Tumour", "OK", f"Mean {m_adc:.0f} -> Necrosis/Cyst likely")
                    else:
                        self.log("CLINICAL", "ADC Tumour", "OK", f"Mean {m_adc:.0f}")
            else:
                self.log(
                    "CLINICAL",
                    "ADC Tumour",
                    "WARN",
                    f"Shape mismatch ({adc_img.shape} vs SEG {seg_img.shape}) — skipping check",
                )

    # ─────────────────────────────────────────────────────────────────
    # FINISH
    # ─────────────────────────────────────────────────────────────────
    def finish(self):
        banner("FINAL SUMMARY")
        s = self.summary
        print(f"  Total Checks : {s['total']}")
        print(f"  {C.GREEN}Passed       : {s['OK']}{C.RESET}")
        print(f"  {C.YELLOW}Warnings     : {s['WARN']}{C.RESET}")
        print(f"  {C.RED}Errors       : {s['ERROR']}{C.RESET}")

        # Save JSON report
        report_path = self.output_dir / "preflight_report.json"
        with open(report_path, "w") as f:
            json.dump(
                {"summary": self.summary, "checks": self.results, "timestamp": datetime.now().isoformat()}, f, indent=2
            )

        print(f"\n  Detailed report saved → {report_path.name}\n")

        if s["ERROR"] > 0:
            print(f"{C.RED}{C.BOLD}❌ Preflight failed with {s['ERROR']} critical errors.{C.RESET}")
            sys.exit(1)
        else:
            print(f"{C.GREEN}{C.BOLD}✅ Preflight passed.{C.RESET}")
            sys.exit(0)


if __name__ == "__main__":
    p = Preflight()
    p.check_layer1()
    p.check_layer2()
    p.check_layer3()
    p.check_layer4()
    p.check_layer5()
    p.finish()
