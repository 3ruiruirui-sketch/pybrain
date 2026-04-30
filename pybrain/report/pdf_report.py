# pybrain/report/pdf_report.py
"""PDF report generation for clinical research use."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def generate_report(
    output_dir: Path,
    patient: Dict[str, Any],
    quality: Dict[str, Any],
    locale: str = "en",
) -> Optional[Path]:
    """
    Generate a PDF research report.

    This is a stub — the full implementation uses ReportLab.
    Returns the path to the generated PDF, or None on failure.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

        pdf_path = output_dir / f"report_{locale}.pdf"
        c = canvas.Canvas(str(pdf_path), pagesize=A4)
        c.drawString(100, 750, "PY-BRAIN Research Report")
        c.drawString(100, 730, f"Patient: {patient.get('name', 'Unknown')}")
        c.drawString(100, 710, f"Age: {patient.get('age', '?')}  Sex: {patient.get('sex', '?')}")

        vols = quality.get("volumes_cc", {})
        y = 680
        for region, cc in vols.items():
            c.drawString(100, y, f"{region}: {cc:.1f} cc")
            y -= 20

        # Molecular prediction section
        import json as _json

        mol_path = output_dir / "molecular_prediction.json"
        if mol_path.exists():
            try:
                with open(mol_path) as f:
                    mol = _json.load(f)

                y -= 10
                c.setFont("Helvetica-Bold", 11)
                c.drawString(100, y, "Molecular Status Prediction (Imaging-Based)")
                c.setFont("Helvetica", 9)
                y -= 20

                idh = mol.get("idh", {})
                idh_text = (
                    f"IDH Status: {idh.get('prediction', 'unknown').upper()} "
                    f"(probability: {idh.get('probability', 0):.0%}, "
                    f"confidence: {idh.get('confidence_level', 'low')})"
                )
                c.drawString(100, y, idh_text)
                y -= 18

                mgmt = mol.get("mgmt", {})
                mgmt_text = (
                    f"MGMT Methylation: {mgmt.get('prediction', 'unknown').upper()} "
                    f"(probability: {mgmt.get('probability', 0):.0%}, "
                    f"confidence: {mgmt.get('confidence_level', 'low')})"
                )
                c.drawString(100, y, mgmt_text)
                y -= 18

                disclaimer = mol.get("disclaimer", "")
                if disclaimer:
                    c.setFont("Helvetica-Oblique", 7)
                    c.drawString(100, y, disclaimer[:90])
                    y -= 12
                    if len(disclaimer) > 90:
                        c.drawString(100, y, disclaimer[90:180])
            except Exception:
                pass

        c.drawString(100, y - 20, "RESEARCH USE ONLY — NOT FOR CLINICAL DIAGNOSIS")
        c.save()
        logger.info(f"Report saved: {pdf_path}")
        return pdf_path
    except ImportError:
        logger.warning("ReportLab not installed — skipping PDF report")
        return None
    except Exception as exc:
        logger.warning(f"PDF generation failed: {exc}")
        return None
