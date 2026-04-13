#!/usr/bin/env python3
"""
Script de debug visual para inspeção dos outputs do pipeline PY-BRAIN.
Gera figura 2x2 comparando FLAIR, segmentação, incerteza MC-Dropout e mean_prob.
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

def find_tumor_slice(seg_volume, axis=2):
    """Encontrar slice axial com maior volume de tumor."""
    # seg_volume: shape (H, W, D)
    tumor_voxels_per_slice = np.sum(seg_volume > 0, axis=(0, 1))
    return np.argmax(tumor_voxels_per_slice)

def create_debug_visualization():
    # Caminhos
    results_dir = Path("/Users/ssoares/Downloads/PY-BRAIN/results/debug_session_20260413_020415")
    monai_dir = Path("/Users/ssoares/Downloads/PY-BRAIN/nifti/monai_ready/brats_test")
    
    output_png = results_dir / "debug_visual_braTS2021_00000.png"
    
    print("=" * 70)
    print("DEBUG VISUAL - PY-BRAIN Pipeline Outputs")
    print("=" * 70)
    
    # 1) Carregar imagens
    print("\n📁 Carregando ficheiros...")
    
    # FLAIR como referência anatómica
    flair_path = monai_dir / "flair.nii.gz"
    if not flair_path.exists():
        # Fallback: tentar sem .nii.gz
        flair_path = monai_dir / "FLAIR_resampled.nii.gz"
    
    seg_path = results_dir / "segmentation_full.nii.gz"
    unc_path = results_dir / "mc_dropout_segresnet_uncertainty.nii.gz"
    prob_path = results_dir / "mc_dropout_segresnet_mean_prob.nii.gz"
    
    print(f"  FLAIR: {flair_path}")
    print(f"  Segmentação: {seg_path}")
    print(f"  Incerteza: {unc_path}")
    print(f"  Mean Prob: {prob_path}")
    
    # Carregar
    flair_img = nib.load(str(flair_path))
    seg_img = nib.load(str(seg_path))
    unc_img = nib.load(str(unc_path))
    prob_img = nib.load(str(prob_path))
    
    flair = flair_img.get_fdata()
    seg = seg_img.get_fdata().astype(np.int32)
    
    # Incerteza: shape pode ser (3, H, W, D) ou (H, W, D, 3)
    unc = unc_img.get_fdata()
    prob = prob_img.get_fdata()
    
    print(f"\n📊 Shapes:")
    print(f"  FLAIR: {flair.shape}")
    print(f"  Seg: {seg.shape}")
    print(f"  Uncertainty: {unc.shape}")
    print(f"  Mean Prob: {prob.shape}")
    
    # 2) Extrair canais WT (Whole Tumor)
    # Segmentation: 1=necrotic, 2=edema, 4=enhancing → WT = 1+2+4
    wt_mask = (seg > 0).astype(np.float32)
    
    # Uncertainty e Prob: assumir formato (3, H, W, D) canais-first
    if unc.shape[0] == 3:
        unc_wt = unc[0]  # Channel 0 = WT
        prob_wt = prob[0]
    elif unc.shape[-1] == 3:
        unc_wt = unc[..., 0]
        prob_wt = prob[..., 0]
    else:
        # Fallback: usar média dos canais
        unc_wt = np.mean(unc, axis=0) if unc.shape[0] == 3 else unc
        prob_wt = np.mean(prob, axis=0) if prob.shape[0] == 3 else prob
    
    print(f"  Uncertainty WT shape: {unc_wt.shape}")
    print(f"  Mean Prob WT shape: {prob_wt.shape}")
    
    # 3) Encontrar slice central com tumor
    slice_idx = find_tumor_slice(seg, axis=2)
    print(f"\n🔍 Slice selecionada (máx tumor): {slice_idx} / {seg.shape[2]}")
    
    # Estatísticas da slice
    wt_voxels = np.sum(wt_mask[:, :, slice_idx])
    print(f"   Voxels WT nesta slice: {wt_voxels}")
    
    # 4) Criar figura
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Debug Visual - BraTS2021_00000 | Slice Axial {slice_idx}\n" + 
                 f"Dice WT=0.93 | MC-Dropout Uncertainty Mean={unc_wt.mean():.4f}", 
                 fontsize=14, fontweight='bold')
    
    # (a) FLAIR original
    ax = axes[0, 0]
    im1 = ax.imshow(flair[:, :, slice_idx], cmap='gray', origin='lower')
    ax.set_title("(a) FLAIR (referência anatómica)", fontweight='bold')
    ax.axis('off')
    
    # (b) FLAIR + contorno segmentação
    ax = axes[0, 1]
    ax.imshow(flair[:, :, slice_idx], cmap='gray', origin='lower', alpha=0.7)
    
    # Criar máscaras coloridas para cada sub-região
    necrotic = (seg[:, :, slice_idx] == 1).astype(float)
    edema = (seg[:, :, slice_idx] == 2).astype(float)
    enhancing = (seg[:, :, slice_idx] == 4).astype(float)
    
    # Overlay com transparência
    from matplotlib.colors import ListedColormap
    cmap_necrotic = ListedColormap(['none', 'red'])
    cmap_edema = ListedColormap(['none', 'yellow'])
    cmap_enhancing = ListedColormap(['none', 'green'])
    
    ax.imshow(necrotic, cmap=cmap_necrotic, origin='lower', alpha=0.6)
    ax.imshow(edema, cmap=cmap_edema, origin='lower', alpha=0.4)
    ax.imshow(enhancing, cmap=cmap_enhancing, origin='lower', alpha=0.6)
    
    ax.set_title("(b) Segmentação sobre FLAIR\nVermelho=Necrótico, Amarelo=Edema, Verde=Enhancing", 
                 fontweight='bold')
    ax.axis('off')
    
    # Legend
    legend_elements = [
        Patch(facecolor='red', alpha=0.6, label='Necrótico'),
        Patch(facecolor='yellow', alpha=0.4, label='Edema'),
        Patch(facecolor='green', alpha=0.6, label='Enhancing')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # (c) Mapa de incerteza MC-Dropout
    ax = axes[1, 0]
    # Normalizar para melhor visualização
    unc_slice = unc_wt[:, :, slice_idx]
    vmax = min(unc_slice.max(), 0.1)  # Cap em 0.1 para visualização
    im3 = ax.imshow(unc_slice, cmap='hot', origin='lower', vmin=0, vmax=vmax)
    ax.set_title(f"(c) MC-Dropout Uncertainty (WT)\nMean={unc_slice.mean():.5f}, Max={unc_slice.max():.4f}", 
                 fontweight='bold')
    ax.axis('off')
    plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04, label='Std Dev')
    
    # (d) Mean Probability
    ax = axes[1, 1]
    prob_slice = prob_wt[:, :, slice_idx]
    im4 = ax.imshow(prob_slice, cmap='viridis', origin='lower', vmin=0, vmax=1)
    ax.set_title(f"(d) MC-Dropout Mean Probability (WT)\nMean={prob_slice.mean():.3f}", 
                 fontweight='bold')
    ax.axis('off')
    plt.colorbar(im4, ax=ax, fraction=0.046, pad=0.04, label='Probability [0,1]')
    
    # Adicionar linha de contorno WT no mean_prob para comparação
    from skimage import measure
    contours = measure.find_contours(wt_mask[:, :, slice_idx], 0.5)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2, alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches='tight', facecolor='black')
    print(f"\n✅ Figura guardada: {output_png}")
    
    # 5) Análise estatística da incerteza
    print("\n📊 Análise de Incerteza na Slice:")
    
    # Dentro vs fora do tumor
    inside_tumor = unc_slice[wt_mask[:, :, slice_idx] > 0]
    outside_tumor = unc_slice[wt_mask[:, :, slice_idx] == 0]
    
    if len(inside_tumor) > 0:
        print(f"  Dentro do WT:   mean={inside_tumor.mean():.5f}, std={inside_tumor.std():.5f}")
    if len(outside_tumor) > 0:
        print(f"  Fora do WT:     mean={outside_tumor.mean():.5f}, std={outside_tumor.std():.5f}")
    
    if len(inside_tumor) > 0 and len(outside_tumor) > 0:
        ratio = inside_tumor.mean() / (outside_tumor.mean() + 1e-8)
        print(f"  Ratio (dentro/fora): {ratio:.2f}x")
    
    print("\n" + "=" * 70)
    print("Análise completa. Ver figura para interpretação visual.")
    print("=" * 70)

if __name__ == "__main__":
    create_debug_visualization()
