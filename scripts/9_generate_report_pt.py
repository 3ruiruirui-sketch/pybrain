#!/usr/bin/env python3
"""
Brain Tumour Analysis — Clinical PDF Report Generator
=======================================================
Generates a structured PDF report combining:
  - Patient information
  - AI segmentation volumes vs radiologist reference
  - MRI visualization images (axial / coronal / sagittal)
  - Extra sequence findings (T2*, DWI, ADC, CT)
  - Validation metrics (if ground_truth.nii.gz exists)
  - Clinical interpretation notes

Requirements:
  pip install reportlab nibabel numpy

Run:
  python3 generate_report.py
  → saves: /Users/ssoares/documents/celeste/results/report_TIMESTAMP.pdf
"""

import sys
import json
import os
import typing
from pathlib import Path
# ── PY-BRAIN session loader ──────────────────────────────────────────
import sys as _sys
from pathlib import Path
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Try both forms of import to satisfy different environments/linters
try:
    from session_loader import get_session, get_paths, get_patient  # type: ignore
except ImportError:
    from scripts.session_loader import get_session, get_paths, get_patient  # type: ignore
try:
    _sess = get_session()
    _paths = get_paths(_sess)
    PATIENT       = get_patient(_sess)
    DICOM_MRI_DIR = _paths["mri_dicom_dir"]
    DICOM_CT_DIR  = _paths.get("ct_dicom_dir")
    NIFTI_DIR     = _paths.get("nifti_dir", _paths["monai_dir"].parent)
    MONAI_DIR     = _paths["monai_dir"]
    EXTRA_DIR     = _paths["extra_dir"]
    BUNDLE_DIR    = _paths["bundle_dir"]
    RESULTS_DIR   = _paths["results_dir"]
    GROUND_TRUTH  = _paths["ground_truth"]
    OUTPUT_DIR    = _paths.get("output_dir", RESULTS_DIR)
    
    DEVICE = "cpu"
    MODEL_DEVICE = "cpu"
    try:
        import torch  # type: ignore
        if torch.backends.mps.is_available():
            DEVICE = torch.device("mps")
            MODEL_DEVICE = torch.device("cpu")
        elif torch.cuda.is_available():
            DEVICE = MODEL_DEVICE = torch.device("cuda")
        else:
            DEVICE = MODEL_DEVICE = torch.device("cpu")
    except ImportError:
        pass

    WT_THRESH, TC_THRESH, ET_THRESH = 0.30, 0.35, 0.40
    HU_CALCIFICATION_LOW, HU_CALCIFICATION_HIGH = 130, 1000
    HU_HAEMORRHAGE_LOW, HU_HAEMORRHAGE_HIGH = 50, 90
    HU_TUMOUR_LOW, HU_TUMOUR_HIGH = 25, 60
    LABEL_NAMES = {1: "Necrotic core", 2: "Edema", 3: "Enhancing tumor"}
    LABEL_COLORS = {1: "Blues", 2: "Greens", 3: "Reds"}
    LABEL_HEX = {1: "#4488ff", 2: "#44cc44", 3: "#ff4444"}
    LABEL_RGB = {1: [0.27, 0.53, 1.00], 2: [0.27, 0.87, 0.27], 3: [1.00, 0.27, 0.27]}
    MRI_SEQUENCE_MAP = {}
    CT_SEQUENCE_MAP = {}
except SystemExit:
    raise
except Exception as e:
    print(f"❌ Failed to load session: {e}")
    _sys.exit(1)
# ─────────────────────────────────────────────────────────────────────

from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────
# ⚙️  PATHS
# ─────────────────────────────────────────────────────────────────────────

RESULTS_BASE  = RESULTS_DIR
GROUND_TRUTH  = RESULTS_DIR / "ground_truth.nii.gz"
TIMESTAMP     = datetime.now().strftime("%Y%m%d_%H%M%S")
REPORT_OUT    = RESULTS_DIR / f"relatorio_{TIMESTAMP}.pdf"

# ──────────────────────────────────────────# Patient template — used IF automated session info is missing
PATIENT_TEMPLATE = {
    "name":       "Maria Celeste Coelho Correia Soares",
    "dob":        "17-02-1945",
    "age":        "81 anos",
    "ref":        "HPA630822",
    "exam_date":  "26-02-2026",
    "institution":"Hospital Particular do Algarve (Faro - Gambelas)",
    "radiologist":"Dr. Ricardo Cruz Martins / Bruno Moreira (O.M. 42231)",
    "exam_type":  "RM Cranio-Encefalico + TC Cranio",
}

# Patient info — prioritize session data, fallback to template
try:
    PATIENT = get_patient(_sess)
except Exception:
    PATIENT = PATIENT_TEMPLATE
# ─────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────

# ReportLab is required for PDF generation
from reportlab.lib.pagesizes import A4  # type: ignore
from reportlab.lib.units import cm, mm  # type: ignore
from reportlab.lib import colors  # type: ignore
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # type: ignore
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY  # type: ignore
from reportlab.platypus import (  # type: ignore
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image, PageBreak, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Rect, String  # type: ignore
from reportlab.graphics import renderPDF  # type: ignore

# ─────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────

PAGE_W, PAGE_H = A4
MARGIN         = 2.0 * cm

def build_styles():
    base = getSampleStyleSheet()
    styles = {}

    styles["title"] = ParagraphStyle(
        "title", fontSize=16, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1a1a2e"), spaceAfter=4,
        alignment=TA_CENTER
    )
    styles["subtitle"] = ParagraphStyle(
        "subtitle", fontSize=10, fontName="Helvetica",
        textColor=colors.HexColor("#555555"), spaceAfter=2,
        alignment=TA_CENTER
    )
    styles["section"] = ParagraphStyle(
        "section", fontSize=11, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#c8553d"),
        spaceBefore=14, spaceAfter=6,
        borderPad=4
    )
    styles["body"] = ParagraphStyle(
        "body", fontSize=9, fontName="Helvetica",
        textColor=colors.HexColor("#222222"),
        leading=14, spaceAfter=4, alignment=TA_JUSTIFY
    )
    styles["body_bold"] = ParagraphStyle(
        "body_bold", fontSize=9, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#222222"), leading=14
    )
    styles["body_small"] = ParagraphStyle(
        "body_small", fontSize=8, fontName="Helvetica",
        textColor=colors.HexColor("#444444"), leading=11,
        spaceAfter=3, alignment=TA_JUSTIFY
    )
    styles["small"] = ParagraphStyle(
        "small", fontSize=7.5, fontName="Helvetica",
        textColor=colors.HexColor("#777777"), leading=11
    )
    styles["disclaimer"] = ParagraphStyle(
        "disclaimer", fontSize=7, fontName="Helvetica-Oblique",
        textColor=colors.HexColor("#999999"),
        leading=10, alignment=TA_CENTER, spaceBefore=8
    )
    styles["metric_good"] = ParagraphStyle(
        "metric_good", fontSize=9, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#2e7d32")
    )
    styles["metric_warn"] = ParagraphStyle(
        "metric_warn", fontSize=9, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#e65100")
    )
    styles["metric_bad"] = ParagraphStyle(
        "metric_bad", fontSize=9, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#c62828")
    )
    return styles


# ─────────────────────────────────────────────────────────────────────────
# TABLE HELPERS
# ─────────────────────────────────────────────────────────────────────────

def info_table(rows, col_widths=None):
    """Two-column label/value table."""
    w = col_widths or [5*cm, 11*cm]
    t = Table(rows, colWidths=w)
    t.setStyle(TableStyle([
        ("FONTNAME",    (0,0),(-1,-1), "Helvetica"),
        ("FONTNAME",    (0,0),(0,-1),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0),(-1,-1), 9),
        ("TEXTCOLOR",   (0,0),(0,-1),  colors.HexColor("#555555")),
        ("TEXTCOLOR",   (1,0),(1,-1),  colors.HexColor("#111111")),
        ("BOTTOMPADDING",(0,0),(-1,-1),3),
        ("TOPPADDING",  (0,0),(-1,-1),3),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),
         [colors.HexColor("#f9f9f9"), colors.white]),
        ("LINEBELOW",   (0,-1),(-1,-1),
         0.5, colors.HexColor("#dddddd")),
    ]))
    return t


def volume_table(rows, headers):
    """Coloured volume comparison table."""
    all_rows = [headers] + rows
    col_w    = [(PAGE_W - 2*MARGIN) / len(headers)] * len(headers)
    t        = Table(all_rows, colWidths=col_w)
    t.setStyle(TableStyle([
        # Header
        ("BACKGROUND",   (0,0), (-1,0),  colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",    (0,0), (-1,0),  colors.white),
        ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 9),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
        # Data rows
        ("ROWBACKGROUNDS",(0,1),(-1,-1),
         [colors.HexColor("#f4f4f4"), colors.white]),
        ("GRID",         (0,0), (-1,-1),
         0.5, colors.HexColor("#cccccc")),
        # Highlight whole tumour row
        ("BACKGROUND",   (0,1), (-1,1),  colors.HexColor("#fff3e0")),
        ("FONTNAME",     (0,1), (-1,1),  "Helvetica-Bold"),
    ]))
    return t


def metrics_table(metrics: dict):
    """Validation metrics table with colour-coded grades."""
    def grade_color(metric, value):
        if metric == "hd95_mm":
            if value < 5:   return colors.HexColor("#2e7d32")
            if value < 15:  return colors.HexColor("#e65100")
            return colors.HexColor("#c62828")
        else:
            thresh_good = 0.80
            thresh_warn = 0.60
            if value >= thresh_good: return colors.HexColor("#2e7d32")
            if value >= thresh_warn: return colors.HexColor("#e65100")
            return colors.HexColor("#c62828")

    def grade_str(metric, value):
        c = grade_color(metric, value)
        if c == colors.HexColor("#2e7d32"): return "BOM"
        if c == colors.HexColor("#e65100"): return "MODERADO"
        return "FRACO"

    headers = ["Metric", "Value", "Threshold", "Grade"]
    rows    = [
        ["Score Dice",         f"{metrics.get('dice',0):.4f}",
         "> 0.80", grade_str("dice", metrics.get("dice",0))],
        ["HD95 (mm)",          f"{metrics.get('hd95_mm',999):.1f} mm",
         "< 5 mm",  grade_str("hd95_mm", metrics.get("hd95_mm",999))],
        ["Sensibilidade",        f"{metrics.get('sensitivity',0):.4f}",
         "> 0.80", grade_str("dice", metrics.get("sensitivity",0))],
        ["Especificidade",        f"{metrics.get('specificity',0):.4f}",
         "> 0.99", ""],
        ["FP volume (cc)",     f"{metrics.get('fp_volume_cc',0):.2f} cc",
         "< 5 cc",  ""],
        ["FN volume (cc)",     f"{metrics.get('fn_volume_cc',0):.2f} cc",
         "",        ""],
        ["Vol. Previsto (cc)", f"{metrics.get('pred_volume_cc',0):.1f} cc",
         "",        ""],
        ["GT vol (cc)",        f"{metrics.get('gt_volume_cc',0):.1f} cc",
         "",        ""],
    ]

    col_w = [6*cm, 3.5*cm, 3.5*cm, 3*cm]
    all_r = [headers] + rows
    t     = Table(all_r, colWidths=col_w)

    style = [
        ("BACKGROUND",    (0,0),(-1,0),  colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",     (0,0),(-1,0),  colors.white),
        ("FONTNAME",      (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 9),
        ("ALIGN",         (1,0),(-1,-1), "CENTER"),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),
         [colors.HexColor("#f9f9f9"), colors.white]),
        ("GRID",          (0,0),(-1,-1),
         0.5, colors.HexColor("#cccccc")),
    ]

    # Colour grade cells
    grade_colors = {
        "BOM":     colors.HexColor("#2e7d32"),
        "MODERADO": colors.HexColor("#e65100"),
        "FRACO":     colors.HexColor("#c62828"),
    }
    for i, row in enumerate(rows):
        grade = row[3]
        if grade in grade_colors:
            style.append(("TEXTCOLOR", (3, i+1), (3, i+1),
                          grade_colors[grade]))
            style.append(("FONTNAME",  (3, i+1), (3, i+1),
                          "Helvetica-Bold"))

    t.setStyle(TableStyle(style))
    return t


# ─────────────────────────────────────────────────────────────────────────
# PAGE TEMPLATE — header and footer on every page
# ─────────────────────────────────────────────────────────────────────────

def make_header_footer(canvas, doc):
    canvas.saveState()
    # Header bar
    canvas.setFillColor(colors.HexColor("#1a1a2e"))
    canvas.rect(0, PAGE_H - 1.5*cm, PAGE_W, 1.5*cm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(MARGIN, PAGE_H - 1.0*cm,
                      "ANÁLISE DE TUMOR CEREBRAL — RELATÓRIO DE INVESTIGAÇÃO IA")
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 1.0*cm,
                           f"Página {doc.page}  |  {datetime.now().strftime('%d/%m/%Y')}")

    # Footer
    canvas.setFillColor(colors.HexColor("#eeeeee"))
    canvas.rect(0, 0, PAGE_W, 1.2*cm, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.setFont("Helvetica-Oblique", 6.5)
    canvas.drawCentredString(
        PAGE_W / 2, 0.45*cm,
        "APENAS PARA FINS DE INVESTIGAÇÃO — NÃO É UM RELATÓRIO DE DIAGNÓSTICO CLÍNICO — "
        "Todos os achados devem ser verificados por um médico neurorradiologista"
    )
    canvas.restoreState()


# ─────────────────────────────────────────────────────────────────────────
# MAIN REPORT BUILDER
# ─────────────────────────────────────────────────────────────────────────

def build_report():
    # Use output directory from session
    latest_dir = Path(OUTPUT_DIR)
    if not latest_dir.exists():
        print(f"❌ Output directory not found: {latest_dir}")
        sys.exit(1)
    print(f"  Using results: {latest_dir.name}")

    # Load primary tumour statistics (Standard or Ensemble)
    stats_path = latest_dir / "tumor_stats_ensemble.json"
    if not stats_path.exists():
        stats_path = latest_dir / "tumor_stats.json"
    
    stats: dict[str, typing.Any] = {}
    if stats_path.exists():
        try:
            with open(stats_path) as f:
                stats = json.load(f)
            print(f"  Loaded stats from {stats_path.name}")
        except json.JSONDecodeError:
            print(f"  ⚠️ Error parsing {stats_path.name} — skipping.")
            stats = {}

    # Load location analysis
    location: dict[str, typing.Any] = {}
    loc_path = latest_dir / "tumour_location.json"
    if loc_path.exists():
        try:
            with open(loc_path) as f:
                location = json.load(f)
        except json.JSONDecodeError:
            print(f"  ⚠️ Error parsing {loc_path.name} — skipping.")
            location = {}

    # Load morphology analysis
    morphology: dict[str, typing.Any] = {}
    morph_path = latest_dir / "morphology.json"
    if morph_path.exists():
        try:
            with open(morph_path) as f:
                morphology = json.load(f)
        except json.JSONDecodeError:
            print(f"  ⚠️ Error parsing {morph_path.name} — skipping.")
            morphology = {}

    # Load radiomics features and ML classification
    radiomics: dict[str, typing.Any] = {}
    rad_path = latest_dir / "radiomics_features.json"
    if rad_path.exists():
        try:
            with open(rad_path) as f:
                radiomics = json.load(f)
        except json.JSONDecodeError:
            print(f"  ⚠️ Error parsing {rad_path.name} — skipping.")
            radiomics = {}
            
    # Load WHO metrics (calcification etc)
    who_metrics = radiomics.get("who_metrics", {})
    ct_calc = who_metrics.get("ct_calcification_cc", 0.0)
            
    rad_report_text = ""
    rad_txt_path = latest_dir / "radiomics_report.txt"
    if rad_txt_path.exists():
        with open(rad_txt_path) as f:
            rad_report_text = f.read()

    # Collect image files
    images = {
        "axial":    latest_dir / "view_axial.png",
        "coronal":  latest_dir / "view_coronal.png",
        "sagittal": latest_dir / "view_sagittal.png",
    }

    # Collect validation metrics
    # Also check latest validation JSON
    val_files = sorted(RESULTS_BASE.glob("validation_*.json"))
    val_metrics = stats.get("validation", None) # Initialize from stats
    if val_files and not val_metrics: # Only try to load from file if not already in stats
        try:
            with open(val_files[-1]) as f:
                val_data = json.load(f)
            if val_data.get("metrics"):
                val_metrics = val_data["metrics"][0]
        except (json.JSONDecodeError, IndexError):
            print(f"  ⚠️ Error parsing validation file — skipping.")
            val_metrics = None

    ensemble_stats: dict[str, typing.Any] = {}
    ensemble_stats_path = latest_dir / "tumor_stats_ensemble.json"
    if ensemble_stats_path.exists():
        try:
            with open(ensemble_stats_path) as f:
                ensemble_stats = json.load(f)
        except json.JSONDecodeError:
            print(f"  ⚠️ Error parsing {ensemble_stats_path.name} — skipping.")
            ensemble_stats = {}

    REPORT_OUT = latest_dir / f"relatorio_{TIMESTAMP}.pdf"

    # ── Build PDF ──────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        str(REPORT_OUT),
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=2.2*cm, bottomMargin=1.8*cm,
        title="Brain Tumour AI Analysis Report",
        author="Brain Tumour Analysis Pipeline",
        subject=PATIENT["name"],
    )

    S      = build_styles()
    story  = []
    HR     = HRFlowable(width="100%", thickness=0.5,
                        color=colors.HexColor("#dddddd"))

    # Build Patient Info (Session-aware)
    patient_info = PATIENT_TEMPLATE.copy()
    if PATIENT and isinstance(PATIENT, dict):
        for k in patient_info.keys():
            if k in PATIENT and PATIENT[k] and PATIENT[k] != "Unknown":
                patient_info[k] = PATIENT[k]

    # ── PAGE 1: HEADER + PATIENT INFO + VOLUMES ────────────────────

    # Title block
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Relatório de Análise de Tumor Cerebral (Soares)", S["title"]))
    story.append(Paragraph(
        "Segmentação Assistida por IA — Apenas para Investigação", S["subtitle"]))
    story.append(Spacer(1, 0.3*cm))
    story.append(HR)
    story.append(Spacer(1, 0.3*cm))

    # Patient info table
    story.append(Paragraph("Informação do Paciente", S["section"]))
    story.append(info_table([
        ["Paciente",         patient_info["name"]],
        ["Data de Nascimento",   f"{patient_info['dob']}  (Age: {patient_info['age']})"],
        ["Nº de Referência",   patient_info["ref"]],
        ["Data do Exame",       patient_info["exam_date"]],
        ["Instituição",     patient_info["institution"]],
        ["Radiologist",     patient_info["radiologist"]],
        ["Exam Type",       patient_info["exam_type"]],
        ["Data de Análise",   datetime.now().strftime("%d/%m/%Y %H:%M")],
        ["AI Engine",       f"MONAI BraTS Bundle (SegResNet) — "
                            f"{stats.get('segmentation_source','bundle').upper()}"],
    ]))
    story.append(Spacer(1, 0.4*cm))

    # Radiologist findings summary (Dynamic from Session Template)
    story.append(Paragraph("Achados do Radiologista (Referência)", S["section"]))
    findings_text = PATIENT.get("radiologist_findings", "No baseline findings provided.")
    story.append(Paragraph(
        "Referência clínica direta utilizada para alinhamento da IA:",
        S["body"]
    ))
    story.append(Spacer(1, 0.2*cm))
    
    # Session-aware findings display
    findings_rows = [
        ["Base Clínica",  Paragraph(findings_text, S["body_small"])],
        ["Volume Ref.",     f"{PATIENT.get('radiologist_volume_cc', 32.0):.1f} cc"],
        ["Área Alvo",     "Left fronto-parietal/rolandic & Right frontal"],
        ["Mass Effect",     "Gyral swelling, lateral ventricle compression"],
    ]
    story.append(info_table(findings_rows))
    story.append(Spacer(1, 0.4*cm))

    # Tumour volume table
    story.append(Paragraph("Medições de Volume Tumoral (IA)", S["section"]))

    vol    = stats.get("volume_cc", {})
    v_w    = vol.get("whole_tumor",   0.0)
    v_n    = vol.get("necrotic_core", 0.0)
    v_e    = vol.get("edema",         0.0)
    v_en   = vol.get("enhancing",     0.0)
    v_br   = vol.get("brain", stats.get("brain_volume_cc", 0.0))
    v_pct  = stats.get("tumor_pct_brain", (v_w / v_br * 100) if v_br > 0 else 0.0)
    
    # Uncertainty Highlight Logic (Item 2)
    u_pct = stats.get("uncertainty_metrics", {}).get("high_uncertainty_pct", 0.0)
    u_color = None
    if u_pct > 50: u_color = colors.HexColor("#c62828") # Red
    elif u_pct > 25: u_color = colors.HexColor("#e65100") # Orange

    volume_rows = [
        ["Tumor Completo",     f"{v_w:.1f} cc",
         "~32 cc", f"{abs(v_w-32)/32*100:.0f}% diff"],
        ["  Núcleo Necrótico",  f"{v_n:.1f} cc",  "—", ""],
        ["  Edema Peritumoral", f"{v_e:.1f} cc", "—", ""],
        ["  Tumor com Realce",   f"{v_en:.1f} cc", "—", ""],
        ["  Calcificação (TC)",   f"{ct_calc:.1f} cc", "WHO Ref", "Grade Target"],
        ["Volume Cerebral",     f"{v_br:.0f} cc",  "~1200–1400 cc", ""],
        ["Tumor / Cérebro",   f"{v_pct:.1f} %",  "0.5–10 %", ""],
    ]
    v_table = volume_table(
        volume_rows,
        ["Região", "Volume IA", "Referência", "Diferença"]
    )
    if u_color:
        v_table.setStyle(TableStyle([
            ("TEXTCOLOR", (1,1), (1,1), u_color),   # Highlight whole tumor AI Volume
            ("FONTNAME", (1,1), (1,1), "Helvetica-Bold")
        ]))
    story.append(v_table)
    story.append(Spacer(1, 0.2*cm))

    # Volume match indicator
    diff_pct = abs(v_w - 32.0) / 32.0 * 100
    if diff_pct < 5:
        match_text = (f"<b>Excellent volume match:</b> AI measured {v_w:.1f} cc "
                      f"vs radiologist ~32 cc ({diff_pct:.1f}% difference). "
                      f"Segmentation volume is clinically accurate.")
        match_color = "#2e7d32"
    elif diff_pct < 20:
        match_text = (f"<b>Good volume match:</b> AI measured {v_w:.1f} cc "
                      f"vs radiologist ~32 cc ({diff_pct:.1f}% difference).")
        match_color = "#e65100"
    else:
        match_text = (f"<b>Volume mismatch:</b> AI measured {v_w:.1f} cc "
                      f"vs radiologist ~32 cc ({diff_pct:.1f}% difference). "
                      f"Manual correction recommended.")
        match_color = "#c62828"

    story.append(Paragraph(
        f'<font color="{match_color}">{match_text}</font>', S["body"]
    ))

    # ── LOCATION SECTION ─────────────────────────────────────────────
    if location and "error" not in location:
        story.append(Spacer(1, 0.4*cm))
        story.append(Paragraph("Localização e Ramificação Tumoral", S["section"]))

        loc_rows = [
            ["Hemisfério",        location.get("hemisphere","—")],
            ["Lobo / Região",     location.get("lobes","—")],
            ["Profundidade",             location.get("depth","—")],
            ["Diâmetro Máximo",      f"{float(location.get('max_diameter_mm',0)):.0f} mm  "
                                  f"({float(location.get('max_diameter_mm',0))/10:.1f} cm)"],
            ["Caixa Delimitadora",      f"X={location['bounding_box_mm'].get('x_mm',0):.0f}mm  "
                                  f"Y={location['bounding_box_mm'].get('y_mm',0):.0f}mm  "
                                  f"Z={location['bounding_box_mm'].get('z_mm',0):.0f}mm"
                                  if location.get('bounding_box_mm') else "—"],
            ["Centro (RAS mm)",   f"R={location['centre_ras_mm'][0]}  " # type: ignore
                                  f"A={location['centre_ras_mm'][1]}  " # type: ignore
                                  f"S={location['centre_ras_mm'][2]}" # type: ignore
                                  if location.get('centre_ras_mm') else "—"],
            ["Distância à Linha Média",      f"{location.get('midline_dist_mm','—')} mm"],
            ["Distância aos Ventrículos",   f"{location.get('vent_dist_mm','—')} mm"],
            ["Cruzamento da Linha Média",  "SIM — envolvimento bilateral"
                                  if location.get('crosses_midline') else "Não"],
            ["Dispersão do Edema",     f"{location.get('oedema_span_mm',0):.0f} mm lateral"],
        ]
        story.append(info_table(loc_rows))
        story.append(Spacer(1, 0.3*cm))

        # Eloquent areas
        eloquent = location.get("eloquent_areas", [])
        if eloquent:
            story.append(Paragraph("Eloquent Area Proximity:", S["body_bold"]))
            for e in eloquent:
                prefix = "⚠️  " if any(w in e.lower()
                    for w in ["motor","broca","wernicke","speech"]) else "• "
                story.append(Paragraph(f"{prefix}{e}", S["body"]))

        # Rolandic warning box
        if location.get("is_rolandic"):
            warn_style = ParagraphStyle(
                "warn", fontSize=8.5, fontName="Helvetica-Bold",
                textColor=colors.HexColor("#c62828"),
                backColor=colors.HexColor("#fff3f3"),
                borderColor=colors.HexColor("#c62828"),
                borderWidth=1, borderPad=6,
                spaceBefore=6, spaceAfter=6, leading=13
            )
            story.append(Paragraph(
                "⚠️  ROLANDIC / PERIROLANDIC LOCATION — This tumour is in or "
                "near the primary motor cortex. Surgical planning must include "
                "functional MRI (fMRI) and/or intraoperative neuromonitoring. "
                "Risk of motor deficit (contralateral arm/leg weakness). "
                "Radiotherapy margins must spare the motor cortex.",
                warn_style
            ))

    # ── PAGE 2: VISUALIZATIONS ─────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Visualização Avançada — Comparação Multimodal", S["section"]))
    story.append(Paragraph(
        "Kaggle-style multi-modal grid. Rows: T1, T1c, T2, FLAIR. "
        "Columns: Representative slices through the whole tumor mass.",
        S["body"]
    ))
    story.append(Spacer(1, 0.2*cm))

    img_width = PAGE_W - 2*MARGIN

    # Helper: insert image at correct aspect ratio, with fallback text
    def add_viz_image(path, max_height=None, caption=None):
        """Add image scaled to full page width with correct aspect ratio."""
        if not path.exists():
            story.append(Paragraph(
                f"[Visualisation not available — run Stage 10 to generate {path.name}]",
                S["small"]
            ))
            story.append(Spacer(1, 0.3*cm))
            return
        try:
            try:
                from PIL import Image as PILImage # type: ignore
                with PILImage.open(str(path)) as im:
                    px_w, px_h = im.size
                ratio = px_h / px_w
            except ImportError:
                ratio = 0.65  # safe default if PIL not available
            h = img_width * ratio
            if max_height and h > max_height:
                h = max_height
            if caption:
                story.append(Paragraph(caption, S["small"]))
                story.append(Spacer(1, 0.1*cm))
            story.append(Image(str(path), width=img_width, height=h))
            story.append(Spacer(1, 0.4*cm))
        except Exception as e:
            story.append(Paragraph(f"[Image error: {e}]", S["small"]))
            story.append(Spacer(1, 0.2*cm))

    # 1. 4-Panel MRI (ratio ~0.75 → fills page width × 0.75)
    p4panel = latest_dir / "viz_4panel_mri.png"
    add_viz_image(p4panel)

    # 2. Region Overlay (ratio ~0.62)
    story.append(Paragraph("Sobreposições de Sub-regiões Tumorais (3 Planos)", S["section"]))
    poverlay = latest_dir / "viz_region_overlay.png"
    add_viz_image(
        poverlay,
        caption="�� Núcleo Necrótico   🟢 Edema Peritumoral   🔴 Tumor com Realce"
    )

    story.append(PageBreak())

    # 3. Intensity Distributions (ratio ~0.51)
    story.append(Paragraph("Perfil de Intensidade de Sinal", S["section"]))
    pdist = latest_dir / "viz_intensity_distributions.png"
    if pdist.exists():
        story.append(Paragraph(
            "Spectral histograms and boxplots showing signal characteristics across different MRI sequences. "
            "Helps differentiate edema from necrotic core and solid enhancing tissue.",
            S["body"]
        ))
        story.append(Spacer(1, 0.2*cm))
    add_viz_image(pdist)

    # 4. Confidence Heatmap (ratio ~0.28 — very wide, cap at 7 cm tall)
    story.append(Paragraph("Mapas de Confiança / Probabilidade da IA", S["section"]))
    pattn = latest_dir / "viz_confidence_heatmap.png"
    if not pattn.exists():
        pattn = latest_dir / "viz_attention_heatmap.png"  # backward compatibility
    
    if pattn.exists():
        story.append(Paragraph(
            "Top row: AI confidence heatmap (red = 100% tumor probability). "
            "Bottom row: Final segmentation consensus used for volumetric data.",
            S["body"]
        ))
        story.append(Spacer(1, 0.2*cm))
    add_viz_image(pattn, max_height=7*cm)

    # 5. MRIcroGL 3D Renders
    mricro_dir = latest_dir / "mricrogl_renders"
    if mricro_dir.exists():
        m_renders = list(mricro_dir.glob("*_render_oblique.png")) + \
                    list(mricro_dir.glob("*_glass_oblique.png"))
        if m_renders:
            story.append(Paragraph("Renderização Volumétrica Avançada 3D (MRIcroGL)", S["section"]))
            story.append(Paragraph(
                "High-fidelity ray-casting renders showing the 3D spatial relationship between the "
                "contrast-enhancing core (red) and the peritumoural edema (green/yellow).",
                S["body"]
            ))
            story.append(Spacer(1, 0.2*cm))
            # Show up to 2 key renders
            for i, m_img in enumerate(m_renders):
                if i >= 2: break
                story.append(Image(str(m_img), width=img_width*0.5, height=img_width*0.4))
            story.append(Spacer(1, 0.5*cm))

    # 6. Interactive Links
    story.append(Paragraph("Análise Interativa 3D e por Cortes", S["section"]))
    link_rows = []
    p3d = latest_dir / "viz_3d_surface.html"
    panim = latest_dir / "viz_slice_animation.html"
    
    if p3d.exists():
        link_rows.append(["Visualização de Superfície 3D", "viz_3d_surface.html", "Interactive Plotly mesh (necrotic, edema, enhancing)"])
    if panim.exists():
        link_rows.append(["Animação por Cortes", "viz_slice_animation.html", "Frame-by-frame scrollable viewer (200MB file)"])
    
    if link_rows:
        story.append(Paragraph(
            "The following interactive HTML files are available in the results folder for deeper analysis:",
            S["body"]
        ))
        story.append(Spacer(1, 0.1*cm))
        story.append(info_table(link_rows))

    # ── Original PAGE 2 (Multi-slice view) remains or is skipped? 
    # Let's keep it but simplified.
    story.append(PageBreak())
    story.append(Paragraph("Visualização de RMN — Grelhas Tradicionais", S["section"]))
    story.append(Paragraph(
        "Each grid shows 8 representative slices in the given orientation. "
        "Rows: T1, T1c (post-contrast), FLAIR, T2, Segmentation overlay.",
        S["body"]
    ))
    story.append(Spacer(1, 0.3*cm))

    for view_name, img_path in images.items():
        if img_path.exists():
            story.append(Paragraph(
                f"{view_name.capitalize()} View", S["body_bold"]))
            story.append(Spacer(1, 0.1*cm))
            try:
                img = Image(str(img_path), width=img_width, height=img_width*0.55)
                story.append(img)
            except Exception as e:
                story.append(Paragraph(f"[Image unavailable: {e}]", S["small"]))
            story.append(Spacer(1, 0.3*cm))
        else:
            story.append(Paragraph(
                f"[{view_name.capitalize()} view image not found — "
                f"run brain_tumor_analysis.py first]", S["small"]
            ))

    # ── PAGE 3: EXTRA SEQUENCES + VALIDATION ──────────────────────
    story.append(PageBreak())

    # ML Classification & Radiomics
    if radiomics and "classification" in radiomics:
        c = radiomics["classification"]
        story.append(Paragraph("Classificação por Machine Learning (Experimental)", S["section"]))
        
        ml_rows = []
        if "high_grade" in c:
            hg = c["high_grade"]
            hgp = hg.get("probability")
            hgp_str = f"{hgp*100:.0f}%" if hgp is not None else "N/A"
            ml_rows.append(["Prob. Alto Grau", f"{hgp_str}  ({hg.get('confidence','—')} conf.)"])
        
        if "primary_vs_metastasis" in c:
            pm = c["primary_vs_metastasis"]
            pmp = pm.get("probability_primary_gbm")
            pmp_str = f" ({pmp*100:.0f}% Primary)" if pmp is not None else ""
            ml_rows.append(["Tumour Type", f"{pm.get('most_likely','—')}{pmp_str}"])
            
        if "idh_mutation" in c:
            idh = c["idh_mutation"]
            ml_rows.append(["IDH Status", f"{idh.get('most_likely','—')}  ({idh.get('confidence','—')} conf.)"])
            
        if "mgmt_methylation" in c:
            mgmt = c["mgmt_methylation"]
            mgp = mgmt.get("probability")
            mgp_str = f"Probability: {mgp*100:.0f}%" if mgp is not None else "Status: Unknown"
            ml_rows.append(["MGMT Status", f"{mgp_str}  ({mgmt.get('confidence','—')} conf.)"])
            
        if "aggressiveness" in c:
            agg = c["aggressiveness"]
            ml_rows.append(["Aggressiveness AI", f"{agg.get('score_0_to_10',0):.1f}/10.0  —  {agg.get('category','—')}"])

        if ml_rows:
            story.append(info_table(ml_rows))
            story.append(Spacer(1, 0.2*cm))

            # Interpretations
            if "primary_vs_metastasis" in c and "interpretation" in c["primary_vs_metastasis"]:
                story.append(Paragraph(f"<b>Primary vs Met:</b> {c['primary_vs_metastasis']['interpretation']}", S["body"]))
            if "idh_mutation" in c and "interpretation" in c["idh_mutation"]:
                story.append(Paragraph(f"<b>IDH Mutation:</b> {c['idh_mutation']['interpretation']}", S["body"]))
            if "mgmt_methylation" in c and "interpretation" in c["mgmt_methylation"]:
                story.append(Paragraph(f"<b>MGMT:</b> {c['mgmt_methylation']['interpretation']}", S["body"]))
                
        story.append(Spacer(1, 0.4*cm))

        # ── BrainIAC Genomic Foundation Model ──────────────────────
        if "idh_mutation" in c:
            story.append(Paragraph("BrainIAC™ Genomic Foundation Model (SOTA)", S["section"]))
            story.append(Paragraph(
                "Non-invasive genomic characterization using the BrainIAC Vision Transformer (ViT) "
                "foundation model. This approach moves beyond traditional radiomics by analyzing "
                "higher-order spatial-temporal features from the full 3D multi-modal scan.",
                S["body"]
            ))
            
            idh_data = c["idh_mutation"]
            status = idh_data.get("most_likely", "Unknown")
            confidence = idh_data.get("confidence", "N/A")
            interpret = idh_data.get("interpretation", "No interpretation available.")
            
            story.append(info_table([
                ["Predicted Genotype", f"<b>{status}</b>"],
                ["ML Confidence",      confidence],
                ["Clinical Implication", interpret]
            ]))
            story.append(Spacer(1, 0.4*cm))
        # ─────────────────────────────────────────────────────────────

    # Radiomics Detailed Report
    if rad_report_text:
        story.append(Paragraph("Detailed Radiomics Profile", S["section"]))
        rad_style = ParagraphStyle(
            "rad_mono", fontName="Courier", fontSize=6.5, leading=8.5,
            textColor=colors.HexColor("#333333"), backColor=colors.HexColor("#f8f8f8"),
            borderPadding=6, borderWidth=0.5, borderColor=colors.HexColor("#dddddd"),
            wordWrap='CJK'
        )
        # Format for ReportLab ML
        text_html = rad_report_text.replace("\n", "<br/>")
        story.append(Paragraph(text_html, rad_style))
        story.append(Spacer(1, 0.4*cm))

    # Tumour Morphology
    if morphology and "shape" in morphology:
        story.append(Paragraph("Tumour Morphology", S["section"]))
        sh = morphology["shape"]
        cr = morphology.get("clinical_ratios", {})
        
        morph_rows = [
            ["Sphericity", f"{sh.get('sphericity',0):.3f}  (1.0 = perfect sphere)"],
            ["Surface Area", f"{sh.get('surface_area_mm2',0):.1f} mm²"],
            ["Oedema:Core ratio", f"{cr.get('edema_to_core_ratio',0):.2f}"],
            ["Necrosis fraction", f"{cr.get('necrosis_fraction',0)*100:.1f}% of core"],
            ["Enhancing fraction", f"{cr.get('enhancement_fraction',0)*100:.1f}% of core"],
        ]
        story.append(info_table(morph_rows))
        story.append(Spacer(1, 0.2*cm))
        
        interp = morphology.get("interpretation", [])
        for line in interp:
            story.append(Paragraph(f"• {line}", S["body"]))
            
        story.append(Spacer(1, 0.4*cm))

        # Add morphology image
        morph_img_path = latest_dir / "morphology_analysis.png"
        if morph_img_path.exists():
            try:
                story.append(Paragraph("Morphology & ADC Histogram Dashboard", S["body_bold"]))
                story.append(Spacer(1, 0.1*cm))
                img_width = PAGE_W - 2*MARGIN
                # Figure is 16x10 inches in the script, so height is width * 0.625
                img = Image(str(morph_img_path), width=img_width, height=img_width*0.625)
                story.append(img)
                story.append(Spacer(1, 0.4*cm))
            except Exception as e:
                story.append(Paragraph(f"[Morphology dashboard unavailable: {e}]", S["small"]))

    # Extra sequence findings
    extra = radiomics.get("intensity", {})
    if extra:
        story.append(Paragraph(
            "Extra Sequence Analysis (T2*, DWI, ADC, CT)", S["section"]))
        story.append(Paragraph(
            "Intensity measurements inside the AI-segmented tumour mask "
            "for each extra sequence. Values reflect tissue characteristics "
            "within the segmented region.",
            S["body"]
        ))
        story.append(Spacer(1, 0.2*cm))

        extra_rows = []
        interpretations = {
            "T2star": "Low signal = calcification or haemosiderin deposits "
                      "(confirmed by radiologist report)",
            "DWI":    "Trace-weighted diffusion signal inside tumour",
            "ADC":    "800-1200 = moderate cellularity (mixed viable + necrotic). "
                      "Consistent with high-grade tumour or metastasis.",
            "CT":     "Mean Hounsfield Units inside tumour mask",
            "CT_Calc":"Calcification mask — presence confirmed by CT density > 130 HU",
            "CT_Haem":"Haemorrhage mask — acute blood (HU 50-90)",
            "CT_density": "Hyperdense tumour (HU 25-60) or hypodense cystic region",
        }
        if not isinstance(extra, dict):
            extra = {}
            
        for seq_name, seq_stats in extra.items():
            if seq_name not in interpretations:
                continue
            mean_v = seq_stats.get("mean", 0)
            std_v  = seq_stats.get("std",  0)
            interp = interpretations.get(seq_name, "")
            extra_rows.append([
                seq_name,
                f"{mean_v:.1f} +/- {std_v:.1f}",
                interp
            ])

        if extra_rows:
            t = Table(
                [["Sequence", "Mean +/- Std", "Interpretation"]] + extra_rows,
                colWidths=[2.5*cm, 3*cm, 10.5*cm]
            )
            t.setStyle(TableStyle([
                ("BACKGROUND",   (0,0),(-1,0), colors.HexColor("#1a1a2e")),
                ("TEXTCOLOR",    (0,0),(-1,0), colors.white),
                ("FONTNAME",     (0,0),(-1,0), "Helvetica-Bold"),
                ("FONTSIZE",     (0,0),(-1,-1),8),
                ("ALIGN",        (1,0),(1,-1), "CENTER"),
                ("VALIGN",       (0,0),(-1,-1),"TOP"),
                ("TOPPADDING",   (0,0),(-1,-1),4),
                ("BOTTOMPADDING",(0,0),(-1,-1),4),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),
                 [colors.HexColor("#f9f9f9"), colors.white]),
                ("GRID",         (0,0),(-1,-1),
                 0.5, colors.HexColor("#cccccc")),
                ("WORDWRAP",     (2,1),(2,-1), True),
            ]))
            story.append(t)
        story.append(Spacer(1, 0.4*cm))

    # Validation metrics
    if val_metrics:
        story.append(Paragraph(
            "Segmentation Validation Metrics", S["section"]))
        story.append(Paragraph(
            "Note: This report compares automated AI measurements against a "
            "reference manual assessment. Any discrepancies may indicate manual "
            "correction performed in an external editor.",
            S["body"]
        ))
        story.append(Spacer(1, 0.2*cm))
        story.append(metrics_table(val_metrics))
        story.append(Spacer(1, 0.2*cm))

        # Interpretation
        dice = val_metrics.get("dice", 0)
        if dice >= 0.80:
            interp = ("The AI segmentation shows <b>good agreement</b> with "
                      "the manual ground truth (Dice >= 0.80). The segmentation "
                      "is suitable as a starting point for clinical review.")
        elif dice >= 0.60:
            interp = ("The AI segmentation shows <b>moderate agreement</b> "
                      "with the manual ground truth (Dice 0.60-0.80). "
                      "Manual boundary correction is recommended before "
                      "clinical use.")
        else:
            interp = ("The AI segmentation shows <b>poor agreement</b> with "
                      "the manual ground truth (Dice < 0.60). Significant "
                      "manual correction is required.")
        story.append(Paragraph(interp, S["body"]))
    else:
        story.append(Paragraph("Validation Metrics", S["section"]))
        story.append(Paragraph(
            "Manual correction is recommended for optimal accuracy. You can "
            "(1) correct the segmentation in an external segment editor, "
            "(2) save the volume as segmentation_full.nii.gz, and (3) re-run this report."
            "(3) run validate_segmentation.py.",
            S["body"]
        ))

    # Ensemble Analysis
    if ensemble_stats:
        story.append(PageBreak())
        story.append(Paragraph("Ensemble Analysis — Uncertainty & Attention", S["section"]))
        u_metrics = typing.cast(dict, ensemble_stats.get("uncertainty_metrics", {}))
        
        story.append(info_table([
            ["Mean Uncertainty", f"{u_metrics.get('mean_uncertainty_in_tumor', 0):.3f}"],
            ["High Uncertainty Volume", f"{u_metrics.get('high_uncertainty_volume_cc', 0):.1f} cc"],
            ["High Uncertainty %", f"{u_metrics.get('high_uncertainty_pct', 0):.1f}% of tumor"],
        ]))
        story.append(Spacer(1, 0.4*cm))
        
        u_img = latest_dir / "uncertainty_analysis.png"
        if u_img.exists():
            story.append(Paragraph("Uncertainty Map Analysis", S["body_bold"]))
            story.append(Image(str(u_img), width=PAGE_W - 2*MARGIN, height=(PAGE_W - 2*MARGIN)*0.25))
            story.append(Spacer(1, 0.4*cm))
            
        a_img = latest_dir / "swinunetr_attention_maps.png"
        if a_img.exists():
            story.append(Paragraph("SwinUNETR Attention Map Analysis (Transformer Shift-Window)", S["body_bold"]))
            story.append(Image(str(a_img), width=(PAGE_W - 2*MARGIN)*0.4, height=(PAGE_W - 2*MARGIN)*0.4))
            story.append(Spacer(1, 0.4*cm))
        else:
            story.append(Paragraph("SwinUNETR Attention Maps Not Available", S["body_bold"]))
            story.append(Paragraph(
                "Transformer-based attention maps were not generated for this session because the "
                "SwinUNETR model bundle was not hosted on the current environment. "
                "Uncertainty computation has fallen back to Single-Model boundary entropy.",
                S["body_small"]
            ))
            story.append(Spacer(1, 0.4*cm))

    story.append(Spacer(1, 0.4*cm))

    # Model Summary table
    story.append(Paragraph("AI Model Architecture Summary", S["section"]))
    story.append(Paragraph(
        "O sistema utiliza o MONAI (Medical Open Network for AI) by NVIDIA e pelo King's College London.",
        S["body_small"]
    ))
    story.append(Spacer(1, 0.2*cm))
    
    seg_model_desc = "MONAI BraTS SegResNet"
    seg_model_details = "4.7M parameters — High-fidelity 3D U-Net variant for voxel-wise tumor delineation."
    if ensemble_stats:
        seg_model_desc = "Ensemble: SegResNet + SwinUNETR"
        seg_model_details = "CNN + Transformer Fusion with uncertainty estimation."

    used_gwo = radiomics.get("densewolf_feature_selection_active", False)
    gwo_desc = "Binary Grey Wolf Optimizer (BGWO)" if used_gwo else "Standard Radiomics"
    gwo_details = "DenseWolf-K algorithm selected the most informative radiomics subset." if used_gwo else "Using all raw features for ML prediction."

    ml_consensus_details = "Ensemble prediction based on BGWO-optimized feature clusters." if used_gwo else "Ensemble prediction based on standard radiomics features."
    
    model_rows = [
        ["1. Segmentation", seg_model_desc, seg_model_details],
        ["2. Classification", "XGBoost Classifier", "Trained on BraTS radiomics features for grade/subtype prediction."],
        ["3. Genomics", "BrainIAC Foundation", "87M parameters — Vision Transformer (ViT) for Genomic mutation prediction."],
        ["4. Optimized Features", gwo_desc, gwo_details],
        ["5. ML Consensus", "XGBoost + Consensus voting", ml_consensus_details],
    ]
    t_mod = Table(model_rows, colWidths=[3.5*cm, 4.5*cm, 9*cm])
    t_mod.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("VALIGN",   (0,0), (-1,-1), "TOP"),
        ("TEXTCOLOR", (0,0), (0,-1), colors.HexColor("#1a1a2e")),
        ("FONTNAME", (0,0), (1,-1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("LINEBELOW", (0,0), (-1,-1), 0.5, colors.HexColor("#eeeeee")),
    ]))
    story.append(t_mod)
    story.append(Spacer(1, 0.4*cm))

    # Technical parameters
    story.append(Paragraph("Technical Parameters", S["section"]))
    story.append(info_table([
        ["Main AI Engine",       "Hybrid Pipeline (CNN + XGBoost + Radiomics)"],
        ["Segmentation",        "MONAI BraTS Bundle (SegResNet)"],
        ["Classification",      "DenseNet121 (DL) + XGBoost (ML Consensus)"],
        ["Input modalities",     "T1, T1c, T2, FLAIR (1.0 mm isotropic)"],
        ["Extra sequences",      "T2* (Hemo), DWI/ADC (Cellularity), CT (registered)"],
        ["Inference hardware",   f"{DEVICE} acceleration — {MODEL_DEVICE} fallback"],
        ["Consensus Policy",     "Triple-Engine Ensemble (Veto voting enabled)"],
        ["Report generated",     datetime.now().strftime("%d/%m/%Y %H:%M:%S")],
    ]))

    story.append(Spacer(1, 0.5*cm))

    # Disclaimer
    story.append(HR)
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "IMPORTANT DISCLAIMER: This report was generated by an AI research "
        "pipeline for research and educational purposes only. It does NOT "
        "constitute a medical diagnosis. All AI-generated findings must be "
        "reviewed, validated, and interpreted by a qualified radiologist or "
        "clinician before any clinical decision is made. The AI model "
        "(MONAI BraTS SegResNet) was trained on glioblastoma datasets and "
        "may underperform on atypical lesions, metastases, or cases with "
        "calcifications. Volume measurements carry an estimated uncertainty "
        "of +/- 15% compared to manual expert segmentation.",
        S["disclaimer"]
    ))

    # Build
    doc.build(story, onFirstPage=make_header_footer,
              onLaterPages=make_header_footer)
    return REPORT_OUT


# ─────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────
print("=" * 55)
print("  Brain Tumour PDF Report Generator")
print("=" * 55)

print(f"\n  Output: {REPORT_OUT}\n")

try:
    out = build_report()
    print(f"\n{'='*55}")
    print(f"  Report saved:")
    print(f"  {out}")
    print(f"{'='*55}")
    
    # Automatically open the report (macOS)
    if sys.platform == "darwin":
        os.system(f"open '{out}'")

    print(f"""
  Open the PDF:
    open {out}

  The report includes:
    Page 1  Patient info + radiologist findings + volumes
    Page 2  MRI visualisation grids (axial/coronal/sagittal)
    Page 3  Extra sequences + validation metrics + parameters
""")
except Exception as e:
    import traceback
    print(f"\n❌ Report generation failed: {e}")
    traceback.print_exc()
    print("\nRun:  pip install reportlab")
