"""Clinical QC Flags System — Automatic detection of unreliable cases."""

from dataclasses import dataclass, field
from typing import List
import logging


@dataclass
class ClinicalFlag:
    code: str  # e.g. "LOW_ENHANCEMENT"
    severity: str  # "WARNING" | "CRITICAL"
    message: str
    recommendation: str


@dataclass
class ClinicalQCReport:
    flags: List[ClinicalFlag] = field(default_factory=list)
    overall: str = "OK"  # "OK" | "REVIEW_RECOMMENDED" | "RESULTS_UNRELIABLE"

    def add(self, flag: ClinicalFlag):
        self.flags.append(flag)
        if flag.severity == "CRITICAL":
            self.overall = "RESULTS_UNRELIABLE"
        elif flag.severity == "WARNING" and self.overall == "OK":
            self.overall = "REVIEW_RECOMMENDED"

    def log_all(self, logger):
        if not self.flags:
            logger.info("[QC-FLAGS] ✅ No clinical flags raised.")
            return
        logger.warning(f"[QC-FLAGS] Overall status: {self.overall}")
        for f in self.flags:
            logger.warning(f"[QC-FLAGS] {f.severity} [{f.code}] {f.message} → {f.recommendation}")

    def to_dict(self) -> dict:
        return {"overall": self.overall, "flags": [vars(f) for f in self.flags]}


def evaluate_clinical_flags(
    volumes: dict,
    segmentation: dict,  # {"wt": cc, "tc": cc, "et": cc, "nc": cc}
    mc_dice: float,
    mean_prob: float,
    patient_meta: dict,  # {"age": int, "foci": int, "longitudinal_delta_pct": float}
    logger: logging.Logger,
) -> ClinicalQCReport:

    report = ClinicalQCReport()

    # FLAG 1 — Low enhancement ratio
    wt = segmentation.get("wt", 0)
    et = segmentation.get("et", 0)
    if wt > 0 and (et / wt) < 0.10:
        report.add(
            ClinicalFlag(
                code="LOW_ENHANCEMENT_RATIO",
                severity="CRITICAL",
                message=f"ET/WT = {et / wt:.1%} ({et:.1f}cc / {wt:.1f}cc) — enhancement below 10% of whole tumour",
                recommendation="Verify gadolinium administration and timing. "
                "Consider low-grade or non-enhancing tumour. "
                "Manual radiologist review required.",
            )
        )

    # FLAG 2 — Low model confidence
    if mean_prob < 0.10:
        report.add(
            ClinicalFlag(
                code="LOW_MODEL_CONFIDENCE",
                severity="CRITICAL",
                message=f"Mean tumour probability = {mean_prob:.3f} (threshold: 0.10) — model not activating",
                recommendation="Results are unreliable. Check input normalization, "
                "contrast protocol, and tumour visibility on T1c.",
            )
        )

    # FLAG 3 — Low MC-Dropout agreement
    if mc_dice < 0.50:
        report.add(
            ClinicalFlag(
                code="HIGH_SEGMENTATION_UNCERTAINTY",
                severity="CRITICAL",
                message=f"MC-Dropout Dice = {mc_dice:.4f} — high disagreement across inference samples",
                recommendation="Segmentation boundary is unreliable. Manual delineation recommended.",
            )
        )

    # FLAG 4 — Elderly patient
    age = patient_meta.get("age", 0)
    if age >= 75:
        report.add(
            ClinicalFlag(
                code="ELDERLY_PATIENT",
                severity="WARNING",
                message=f"Patient age {age}y — atypical presentations more common. "
                f"Model trained primarily on adults 18-70y.",
                recommendation="Interpret results with caution. Consider age-related WM changes as confounders.",
            )
        )

    # FLAG 5 — Multi-focal tumour
    foci = patient_meta.get("foci", 1)
    if foci > 1:
        report.add(
            ClinicalFlag(
                code="MULTIFOCAL_TUMOUR",
                severity="WARNING",
                message=f"{foci} distinct foci detected — smallest lesions may be under-segmented",
                recommendation="Review all foci individually. Primary segmentation covers largest focus only.",
            )
        )

    # FLAG 6 — Implausible longitudinal change
    delta = abs(patient_meta.get("longitudinal_delta_pct", 0))
    if delta > 100:
        report.add(
            ClinicalFlag(
                code="IMPLAUSIBLE_VOLUME_CHANGE",
                severity="WARNING",
                message=f"Longitudinal WT delta = {delta:.1f}% — "
                f"change >100% between sessions is clinically implausible",
                recommendation="Verify patient ID, scan date, and prior session data. "
                "Possible registration or session mismatch.",
            )
        )

    return report
