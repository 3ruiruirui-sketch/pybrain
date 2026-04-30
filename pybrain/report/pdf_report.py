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
    longitudinal: Optional[Dict[str, Any]] = None,
    mets: Optional[Dict[str, Any]] = None,
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

        # Longitudinal comparison section
        if longitudinal is not None:
            y -= 30
            c.setFont("Helvetica-Bold", 11)
            c.drawString(100, y, "Longitudinal Comparison")
            c.setFont("Helvetica", 9)
            y -= 20

            # RANO assessment
            rano_response = longitudinal.get("rano_response", "NE")
            c.drawString(100, y, f"RANO Assessment: {rano_response}")
            y -= 15

            # Registration quality
            nmi = longitudinal.get("registration_quality", 0.0)
            c.drawString(100, y, f"Registration Quality (NMI): {nmi:.4f}")
            y -= 15

            # Volume changes table
            volume_changes = longitudinal.get("volume_changes", {})
            if volume_changes:
                y -= 10
                c.setFont("Helvetica-Bold", 9)
                c.drawString(100, y, "Volume Changes:")
                c.setFont("Helvetica", 8)
                y -= 15

                for region, change in volume_changes.items():
                    prior = change.get("prior_cc", 0.0)
                    current = change.get("current_cc", 0.0)
                    pct = change.get("pct_change", 0.0)
                    status = change.get("status", "stable")
                    c.drawString(
                        100, y,
                        f"{region}: {prior:.1f} → {current:.1f} cc ({pct:+.1f}%) [{status}]"
                    )
                    y -= 12

            # Research disclaimer
            y -= 10
            c.setFont("Helvetica-Oblique", 7)
            c.drawString(100, y, "Research RANO interpretation; not for clinical use")
            y -= 12

        # Brain metastases section
        if mets is not None:
            y -= 30
            c.setFont("Helvetica-Bold", 11)
            c.drawString(100, y, "Brain Metastases Analysis")
            c.setFont("Helvetica", 9)
            y -= 20

            # Total lesion count and volume
            total_count = mets.get("total_lesion_count", 0)
            total_volume = mets.get("total_lesion_volume_cc", 0.0)
            c.drawString(100, y, f"Total Lesions: {total_count}")
            y -= 15
            c.drawString(100, y, f"Total Volume: {total_volume:.2f} cc")
            y -= 15

            # Detection and segmentation methods
            detection_method = mets.get("detection_method", "unknown")
            segmentation_method = mets.get("segmentation_method", "unknown")
            c.drawString(100, y, f"Detection: {detection_method}")
            y -= 15
            c.drawString(100, y, f"Segmentation: {segmentation_method}")
            y -= 15

            # Lesion-by-lesion table
            lesions = mets.get("lesions", [])
            if lesions:
                y -= 10
                c.setFont("Helvetica-Bold", 9)
                c.drawString(100, y, "Lesion Details:")
                c.setFont("Helvetica", 8)
                y -= 15

                for lesion in lesions:
                    lesion_id = lesion.get("id", "?")
                    location = lesion.get("location", "Unknown")
                    volume = lesion.get("volume_cc", 0.0)
                    confidence = lesion.get("confidence", 0.0)
                    c.drawString(
                        100, y,
                        f"#{lesion_id}: {location} - {volume:.2f} cc (conf: {confidence:.2f})"
                    )
                    y -= 12

            # Research disclaimer
            y -= 10
            c.setFont("Helvetica-Oblique", 7)
            c.drawString(100, y, "Research mets analysis; not for clinical use")
            y -= 12

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
