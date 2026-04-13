#!/usr/bin/env python3
"""
Análise detalhada do caso BraTS2021_00002 vs 00000.
Investigar por que o Dice WT é significativamente menor (0.70 vs 0.93).
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_case(case_num, results_dir, monai_dir):
    """Analisar um caso BraTS em detalhe."""
    
    print(f"\n{'='*70}")
    print(f"ANÁLISE: BraTS2021_000{case_num}")
    print(f"{'='*70}")
    
    # Carregar ficheiros
    seg_path = results_dir / "segmentation_full.nii.gz"
    gt_path = monai_dir / "ground_truth.nii.gz"
    flair_path = monai_dir / "flair.nii.gz"
    t1c_path = monai_dir / "t1c.nii.gz"
    
    seg = nib.load(str(seg_path)).get_fdata()
    gt = nib.load(str(gt_path)).get_fdata()
    flair = nib.load(str(flair_path)).get_fdata()
    t1c = nib.load(str(t1c_path)).get_fdata()
    
    print(f"Shapes: SEG={seg.shape}, GT={gt.shape}")
    
    # 1) Estatísticas de volume
    seg_wt = (seg > 0).astype(np.float32)
    gt_wt = (gt > 0).astype(np.float32)
    seg_tc = ((seg == 1) | (seg == 4)).astype(np.float32)
    gt_tc = ((gt == 1) | (gt == 4)).astype(np.float32)
    seg_et = (seg == 4).astype(np.float32)
    gt_et = (gt == 4).astype(np.float32)
    
    voxel_vol = 0.001  # cc (1mm³)
    
    print(f"\n📊 VOLUMES (cc):")
    ratio_wt = seg_wt.sum()/gt_wt.sum() if gt_wt.sum() > 0 else 0
    ratio_tc = seg_tc.sum()/gt_tc.sum() if gt_tc.sum() > 0 else 0
    ratio_et = seg_et.sum()/gt_et.sum() if gt_et.sum() > 0 else 0
    print(f"  WT - Pred: {seg_wt.sum() * voxel_vol:.1f}, GT: {gt_wt.sum() * voxel_vol:.1f}, Ratio: {ratio_wt:.2f}")
    print(f"  TC - Pred: {seg_tc.sum() * voxel_vol:.1f}, GT: {gt_tc.sum() * voxel_vol:.1f}, Ratio: {ratio_tc:.2f}")
    print(f"  ET - Pred: {seg_et.sum() * voxel_vol:.1f}, GT: {gt_et.sum() * voxel_vol:.1f}, Ratio: {ratio_et:.2f}")
    
    # 2) Proporções de sub-regiões
    print(f"\n📈 PROPORÇÕES NO GT:")
    gt_total = gt_wt.sum()
    if gt_total > 0:
        print(f"  Necrótico (GT=1):  {(gt == 1).sum() / gt_total * 100:.1f}%")
        print(f"  Edema (GT=2):      {(gt == 2).sum() / gt_total * 100:.1f}%")
        print(f"  Enhancing (GT=4):  {(gt == 4).sum() / gt_total * 100:.1f}%")
    
    # 3) Comparação Pred vs GT
    print(f"\n🎯 DICE SCORES:")
    
    def dice_score(pred, gt):
        intersection = np.sum(pred * gt)
        union = np.sum(pred) + np.sum(gt)
        return 2.0 * intersection / union if union > 0 else 1.0
    
    dice_wt = dice_score(seg_wt, gt_wt)
    dice_tc = dice_score(seg_tc, gt_tc)
    dice_et = dice_score(seg_et, gt_et)
    
    print(f"  WT: {dice_wt:.4f}")
    print(f"  TC: {dice_tc:.4f}")
    print(f"  ET: {dice_et:.4f}")
    
    # 4) Análise de erro
    print(f"\n🔍 ANÁLISE DE ERRO:")
    
    # False positives (prediz tumor onde não há)
    fp = np.sum((seg_wt > 0) & (gt_wt == 0))
    # False negatives (não prediz tumor onde há)
    fn = np.sum((seg_wt == 0) & (gt_wt > 0))
    # True positives
    tp = np.sum((seg_wt > 0) & (gt_wt > 0))
    
    print(f"  True Positives:  {tp} voxels ({tp/gt_wt.sum()*100:.1f}% do GT)")
    print(f"  False Positives: {fp} voxels ({fp/seg_wt.sum()*100:.1f}% da predição)")
    print(f"  False Negatives: {fn} voxels ({fn/gt_wt.sum()*100:.1f}% do GT em falta)")
    
    # 5) Características da imagem
    print(f"\n🖼️  CARACTERÍSTICAS DA IMAGEM:")
    
    # Intensidade no tumor (T1c)
    t1c_in_tumor = t1c[gt_wt > 0]
    t1c_outside = t1c[gt_wt == 0]
    
    print(f"  T1c no tumor:   mean={t1c_in_tumor.mean():.1f}, std={t1c_in_tumor.std():.1f}")
    print(f"  T1c fora:       mean={t1c_outside.mean():.1f}, std={t1c_outside.std():.1f}")
    print(f"  Contraste:      {t1c_in_tumor.mean() - t1c_outside.mean():.1f}")
    
    # FLAIR no edema
    flair_edema = flair[gt == 2]
    flair_brain = flair[(gt == 0) & (flair > 0)]
    
    if len(flair_edema) > 0 and len(flair_brain) > 0:
        print(f"  FLAIR edema:    mean={flair_edema.mean():.1f}")
        print(f"  FLAIR cérebro:  mean={flair_brain.mean():.1f}")
        print(f"  Ratio edema/background: {flair_edema.mean()/flair_brain.mean():.2f}")
    
    return {
        'case': case_num,
        'dice_wt': dice_wt,
        'dice_tc': dice_tc,
        'dice_et': dice_et,
        'vol_wt_pred': seg_wt.sum() * voxel_vol,
        'vol_wt_gt': gt_wt.sum() * voxel_vol,
        'vol_et_gt': gt_et.sum() * voxel_vol,
        'tp_pct': tp / gt_wt.sum() * 100,
        'fn_pct': fn / gt_wt.sum() * 100,
        't1c_contrast': t1c_in_tumor.mean() - t1c_outside.mean()
    }

def compare_cases():
    """Comparar os dois casos lado a lado."""
    
    base_dir = Path("/Users/ssoares/Downloads/PY-BRAIN")
    
    # Caso 00000
    r00 = base_dir / "results/debug_session_20260413_020415"
    m00 = base_dir / "nifti/monai_ready/brats_test"
    stats_00 = analyze_case(0, r00, m00)
    
    # Caso 00002
    r02 = base_dir / "results/validation_BraTS2021_00002"
    m02 = base_dir / "nifti/monai_ready/brats2021_00002"
    stats_02 = analyze_case(2, r02, m02)
    
    # Comparação
    print(f"\n{'='*70}")
    print(f"COMPARAÇÃO: 00000 vs 00002")
    print(f"{'='*70}")
    
    print(f"\n{'Métrica':<30} {'00000':<15} {'00002':<15} {'Diferença'}")
    print(f"{'-'*75}")
    print(f"{'Dice WT':<30} {stats_00['dice_wt']:<15.4f} {stats_02['dice_wt']:<15.4f} " +
          f"{stats_02['dice_wt'] - stats_00['dice_wt']:+.4f}")
    print(f"{'Dice TC':<30} {stats_00['dice_tc']:<15.4f} {stats_02['dice_tc']:<15.4f} " +
          f"{stats_02['dice_tc'] - stats_00['dice_tc']:+.4f}")
    dice_et_diff = stats_02['dice_et'] - stats_00['dice_et']
    print(f"{'Dice ET':<30} {stats_00['dice_et']:<15.4f} {stats_02['dice_et']:<15.4f} {dice_et_diff:+.4f}")
    print(f"{'Volume WT GT (cc)':<30} {stats_00['vol_wt_gt']:<15.1f} {stats_02['vol_wt_gt']:<15.1f} " +
          f"{stats_02['vol_wt_gt'] - stats_00['vol_wt_gt']:+.1f}")
    print(f"{'Volume ET GT (cc)':<30} {stats_00['vol_et_gt']:<15.1f} {stats_02['vol_et_gt']:<15.1f} " +
          f"{stats_02['vol_et_gt'] - stats_00['vol_et_gt']:+.1f}")
    print(f"{'% True Positives':<30} {stats_00['tp_pct']:<15.1f} {stats_02['tp_pct']:<15.1f} " +
          f"{stats_02['tp_pct'] - stats_00['tp_pct']:+.1f}")
    print(f"{'% False Negatives':<30} {stats_00['fn_pct']:<15.1f} {stats_02['fn_pct']:<15.1f} " +
          f"{stats_02['fn_pct'] - stats_00['fn_pct']:+.1f}")
    print(f"{'Contraste T1c':<30} {stats_00['t1c_contrast']:<15.1f} {stats_02['t1c_contrast']:<15.1f} " +
          f"{stats_02['t1c_contrast'] - stats_00['t1c_contrast']:+.1f}")
    
    # Diagnóstico
    print(f"\n{'='*70}")
    print(f"DIAGNÓSTICO")
    print(f"{'='*70}")
    
    findings = []
    
    if stats_02['vol_et_gt'] < stats_00['vol_et_gt'] * 0.5:
        findings.append("🔴 Pouco enhancing no GT (possível necrose predominante)")
    
    if stats_02['fn_pct'] > stats_00['fn_pct'] + 10:
        findings.append("🔴 Pipeline sub-segmenta (muitos false negatives)")
    
    if stats_02['t1c_contrast'] < stats_00['t1c_contrast'] * 0.7:
        findings.append("🟡 Baixo contraste T1c (tumor mal delineado)")
    
    if abs(stats_02['vol_wt_gt'] - stats_02['vol_wt_pred']) > 40:
        findings.append("🔴 Grande diferença de volume (45% no caso 00002)")
    
    if findings:
        print("\nAchados:")
        for f in findings:
            print(f"  {f}")
    else:
        print("\n✅ Sem achados específicos — caso dentro da variação normal")
    
    print(f"\n{'='*70}")

if __name__ == "__main__":
    compare_cases()
