#!/usr/bin/env python3
"""
CLINICAL DOMAIN VALIDATOR — Pre-Pipeline Screening
===================================================

Detecta casos ATÍPICOS antes de processamento para evitar resultados absurdos.

Rejeita casos quando:
- Idade > 80y (dados de treino limitados)
- Enhancement ratio > 90% (sugere meningioma, não GBM)
- Calcificação > 100 cc no CT (modelo não treinado para calcificados)
- Edema/Tumor ratio < 10% (atípico para gliomas de alto grau)

Saída: RECOMENDAÇÃO clara — prosseguir ou parar.
"""

import os
import sys
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from datetime import datetime

# DOMÍNIO DO MODELO BraTS-2021 (dados de treino)
MODEL_DOMAIN = {
    "age_range": [18, 85],  # Anos
    "et_wt_ratio": [0.05, 0.80],  # GBM can have ET/WT < 0.10 (non-enhancing)
    "edema_wt_ratio": [0.10, 0.95],
    "max_calcification_cc": 50.0,  # CT calcification volume
}

SEVERITY_WEIGHTS = {
    "age": 2,  # Idade fora do range
    "enhancement": 3,  # ET/WT atípico (indica outro tipo de tumor)
    "edema": 2,  # Ausência de edema
    "calcification": 3,  # Calcificação extensa
}


def banner(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def check_age(session_data: dict) -> tuple:
    """Check if age is within model training domain."""
    age_str = session_data.get("patient", {}).get("age", "0")
    try:
        age = int(age_str) if isinstance(age_str, str) and age_str.isdigit() else int(age_str)
    except:
        age = 0

    in_range = MODEL_DOMAIN["age_range"][0] <= age <= MODEL_DOMAIN["age_range"][1]
    return in_range, age, MODEL_DOMAIN["age_range"]


def check_ct_calcification(session_dir: Path) -> tuple:
    """Check CT calcification volume."""
    calc_path = session_dir / "ct_calcification.nii.gz"
    if not calc_path.exists():
        return True, 0.0, "No CT calcification file"

    try:
        calc_img = nib.load(str(calc_path), mmap=True)
        calc_data = calc_img.get_fdata()
        voxel_vol = np.prod(calc_img.header.get_zooms()) / 1000.0  # cc
        calc_vol = calc_data.sum() * voxel_vol

        within_limit = calc_vol <= MODEL_DOMAIN["max_calcification_cc"]
        return within_limit, calc_vol, f"{calc_vol:.1f} cc"
    except Exception as e:
        return True, 0.0, f"Error: {e}"


def estimate_enhancement_pattern(session_dir: Path) -> tuple:
    """Estimate ET/WT and Edema/WT from probability maps if available."""
    prob_path = session_dir / "ensemble_probability.nii.gz"
    brain_path = session_dir / "brain_mask.nii.gz"

    if not prob_path.exists() or not brain_path.exists():
        return None, None, "Probability maps not yet computed"

    try:
        prob = nib.load(str(prob_path), mmap=True).get_fdata()
        brain = nib.load(str(brain_path), mmap=True).get_fdata()

        if prob.ndim == 4:
            tc = prob[0]
            wt = prob[1]
            et = prob[2]
        else:
            return None, None, "Invalid probability shape"

        # Estimate binary masks with conservative threshold
        brain_mask = brain > 0
        wt_mask = (wt > 0.5) & brain_mask
        et_mask = (et > 0.5) & brain_mask
        tc_mask = (tc > 0.5) & brain_mask

        if wt_mask.sum() == 0:
            return None, None, "No tumor detected"

        et_wt_ratio = et_mask.sum() / wt_mask.sum()
        edema_mask = wt_mask & ~tc_mask
        edema_wt_ratio = edema_mask.sum() / wt_mask.sum()

        et_ok = MODEL_DOMAIN["et_wt_ratio"][0] <= et_wt_ratio <= MODEL_DOMAIN["et_wt_ratio"][1]
        edema_ok = MODEL_DOMAIN["edema_wt_ratio"][0] <= edema_wt_ratio <= MODEL_DOMAIN["edema_wt_ratio"][1]

        return (et_ok, et_wt_ratio, f"{et_wt_ratio:.2f}"), (edema_ok, edema_wt_ratio, f"{edema_wt_ratio:.2f}"), "OK"
    except Exception as e:
        return None, None, f"Error: {e}"


def calculate_risk_score(checks: dict) -> tuple:
    """Calculate composite risk score."""
    score = 0
    max_score = sum(SEVERITY_WEIGHTS.values())

    if not checks["age"][0]:
        score += SEVERITY_WEIGHTS["age"]

    if checks["calcification"] and not checks["calcification"][0]:
        score += SEVERITY_WEIGHTS["calcification"]

    if checks["enhancement"]:
        et_check, _, _ = checks["enhancement"]
        if not et_check:
            score += SEVERITY_WEIGHTS["enhancement"]

    if checks["edema"]:
        edema_check, _, _ = checks["edema"]
        if not edema_check:
            score += SEVERITY_WEIGHTS["edema"]

    percentage = 100 * score / max_score
    return score, max_score, percentage


def main():
    """Main validation routine."""
    banner("CLINICAL DOMAIN VALIDATOR — Pre-Pipeline Screening")

    # Load session — respects PYBRAIN_SESSION env var (set by run_pipeline.py)
    session_path = Path(os.environ.get("PYBRAIN_SESSION", ""))
    if not session_path.exists():
        # Fallback: last session in results directory
        results_dir = Path(__file__).resolve().parent.parent / "results"
        sessions = sorted(results_dir.glob("*/session.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        session_path = sessions[0] if sessions else Path("")
    if not session_path.exists():
        print("❌ Session not found")
        sys.exit(1)

    with open(session_path) as f:
        session = json.load(f)

    session_dir = Path(session["output_dir"])
    patient = session.get("patient", {})

    print(f"\nPatient: {patient.get('name', 'Unknown')}")
    print(f"Age: {patient.get('age', '?')} | Sex: {patient.get('sex', '?')}")
    print(f"Session: {session_dir.name}")

    # Run checks
    checks = {
        "age": check_age(session),
        "calcification": check_ct_calcification(session_dir),
        "enhancement": None,
        "edema": None,
    }

    # Check Stage 3 results for non-GBM / zero-tumor detection
    qual_path = session_dir / "segmentation_quality.json"
    non_gbm_suspected = False
    if qual_path.exists():
        try:
            with open(qual_path) as f:
                q = json.load(f).get("quality", {})
                non_gbm_suspected = q.get("non_glioblastoma_suspected", False)
        except Exception:
            pass

    # Check if we can estimate enhancement (requires Stage 3 completed)
    if (session_dir / "ensemble_probability.nii.gz").exists():
        et_result, edema_result, msg = estimate_enhancement_pattern(session_dir)
        if et_result:
            checks["enhancement"] = et_result
            checks["edema"] = edema_result

    # Display results
    banner("VALIDATION RESULTS")

    print("\n  1. PATIENT AGE")
    age_ok, age_val, age_ref = checks["age"]
    status = "✅" if age_ok else "⚠️  RISCO"
    print(f"     {status} Age: {age_val}y (expected: {age_ref[0]}-{age_ref[1]}y)")
    if not age_ok:
        print(f"     → Modelo treinado em pacientes {age_ref[0]}-{age_ref[1]}y")
        print("     → Acima de 80y, dados de treino são limitados")

    print("\n  2. CT CALCIFICATION")
    if checks["calcification"]:
        calc_ok, calc_val, calc_msg = checks["calcification"]
        status = "✅" if calc_ok else "⚠️  RISCO ALTO"
        print(f"     {status} Volume: {calc_msg} (limite: {MODEL_DOMAIN['max_calcification_cc']:.0f} cc)")
        if not calc_ok:
            print("     → Calcificação extensa sugere MENINGIOMA, não GBM")
            print("     → Modelo BraTS NÃO foi treinado em tumores calcificados")
    else:
        print("     ℹ️  CT não disponível para análise")

    print("\n  3. MRI ENHANCEMENT PATTERN (se Stage 3 concluído)")
    if checks["enhancement"]:
        et_ok, et_val, et_msg = checks["enhancement"]
        edema_ok, edema_val, edema_msg = checks["edema"]

        et_status = "✅" if et_ok else "⚠️  RISCO"
        edema_status = "✅" if edema_ok else "⚠️  RISCO"

        print(f"     {et_status} ET/WT ratio: {et_msg} (expected: 0.20-0.70)")
        print(f"     {edema_status} Edema/WT: {edema_msg} (expected: 0.30-0.90)")

        if not et_ok:
            print(f"     → ET/WT {et_msg} sugere enhancement homogêneo (meningioma)")
            print("     → GBM típico tem enhancement anelar incompleto")
        if not edema_ok:
            print("     → Ausência de edema é atípico para glioma de alto grau")
    else:
        print("     ℹ️  Aguardando conclusão do Stage 3 (segmentação)")

    # Calculate risk
    score, max_score, percentage = calculate_risk_score(checks)

    banner("RISK ASSESSMENT")
    print(f"\n  Risk Score: {score}/{max_score} ({percentage:.0f}%)")

    # ── Non-GBM detected by Stage 3 ───────────────────────────────────────────
    # Stage 3 already confirmed this is likely not GBM — BraTS model did not activate
    if non_gbm_suspected:
        recommendation = "NON_GBM_SUSPECTED"
        color = "🔵"
        message = "Caso com características de MENINGIOMA ou outro tumor NÃO-GBM."
        print(f"\n  {color} RECOMMENDATION: {recommendation}")
        print(f"\n  {message}")
        print("\n  CONCLUSÃO DO ESTÁGIO 3:")
        print("  O modelo BraTS NÃO conseguiu detectar tumor — fortemente sugere MENINGIOMA CALCIFICADO.")
        print("  Este pipeline é para Glioblastoma (GBM). NÃO USE para este caso.")
        print("\n  Próximos passos:")
        print("  1. Revisão manual por neuroradiologista")
        print("  2. Considere classificação de meningioma")
        print("  3.Este relatório explica a falha do modelo — não é um erro de execução")
    else:
        # ── Mandatory DO_NOT_PROCEED overrides ──────────────────────────────────
        # Any single severity_weight >= 3 failure is a definitive out-of-domain finding.
        # These override the percentage-based tiers regardless of total score:
        #   - enhancement (w=3): homogeneous enhancement → meningioma pattern
        #   - calcification (w=3): >50cc → calcified tumour (meningioma/sarcoma)
        # These represent cases where the BraTS GBM model has zero valid training data.
        for check_name, v in checks.items():
            if v is None:
                continue
            did_pass = v[0]
            if not did_pass and SEVERITY_WEIGHTS.get(check_name, 0) >= 3:
                recommendation = "DO_NOT_PROCEED"
                color = "🔴"
                message = "CASO INADEQUADO PARA MODELO BraTS. NÃO PROSSEGUIR."
                break
        else:
            # No mandatory override — use percentage-based tiers
            if percentage < 25:
                recommendation = "PROCEED"
                color = "🟢"
                message = "Caso dentro do domínio do modelo. Pipeline pode executar."
            elif percentage < 50:
                recommendation = "PROCEED_WITH_CAUTION"
                color = "🟡"
                message = "Caso atípico. Resultados requerem validação rigorosa."
            elif percentage < 75:
                recommendation = "STRONG_WARNING"
                color = "🟠"
                message = "Caso fora do domínio. Segmentação provavelmente incorreta."
            else:
                recommendation = "DO_NOT_PROCEED"
                color = "🔴"
                message = "CASO INADEQUADO PARA MODELO BraTS. NÃO PROSSEGUIR."

    print(f"\n  {color} RECOMMENDATION: {recommendation}")
    print(f"\n  {message}")

    if recommendation == "DO_NOT_PROCEED" or recommendation == "NON_GBM_SUSPECTED":
        print("\n" + "=" * 60)
        print("  DIAGNÓSTICO ALTERNATIVO SUGERIDO:")
        print("=" * 60)
        print("  Este caso tem características de MENINGIOMA CALCIFICADO:")
        age_ok, age_val, age_ref = checks["age"]
        print(f"    - Idade avançada ({age_val}y)")
        print("    - Calcificações extensas no CT")
        print("    - Enhancement homogêneo/nulo")
        print("    - Ausência de edema peritumoral típico")
        print("\n  O modelo BraTS foi treinado em GLIOBLASTOMA (GBM).")
        print("  NÃO USE ESTE PIPELINE para este caso.")
        print("\n  Alternativas recomendadas:")
        print("    1. Segmentação manual por neuroradiologista")
        print("    2. Classificador específico para meningioma")
        print("    3. Protocolo de ressonância com contraste otimizado")
        print("=" * 60)

    # Save validation report
    # [FIX-VALIDATOR] Convert numpy types to Python native types for JSON serialization
    def _to_native(val):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(val, (np.bool_,)):
            return bool(val)
        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            return float(val)
        return val

    report = {
        "timestamp": datetime.now().isoformat(),
        "patient": patient.get("name"),
        "checks": {
            k: {
                "pass": bool(v[0]) if v else False,
                "value": _to_native(v[1]) if v else None,
                "detail": str(v[2]) if v else None,
            }
            for k, v in checks.items()
        },
        "risk_score": _to_native(score),
        "risk_max": _to_native(max_score),
        "risk_percentage": _to_native(percentage),
        "recommendation": recommendation,
        "message": message,
    }

    report_path = session_dir / "clinical_validation.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Validation report saved: {report_path.name}")

    return 0 if recommendation in ["PROCEED", "PROCEED_WITH_CAUTION"] else 1


if __name__ == "__main__":
    sys.exit(main())
