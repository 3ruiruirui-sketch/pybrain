#!/usr/bin/env python3
"""
Preparar caso BraTS2021 para o pipeline PY-BRAIN.
Converte dados BraTS para formato MONAI-ready.
"""

import argparse
import json
import shutil
from pathlib import Path
import nibabel as nib
import numpy as np

def prepare_brats_case(case_id: str, output_dir: Path, brats_root: Path = None):
    """
    Preparar um caso BraTS2021 para o pipeline.
    
    Args:
        case_id: ID do caso (ex: BraTS2021_00000)
        output_dir: Diretório de saída
        brats_root: Raiz dos dados BraTS (default: data/datasets/BraTS2021)
    """
    if brats_root is None:
        brats_root = Path(__file__).parent / "data" / "datasets" / "BraTS2021"
    else:
        brats_root = Path(brats_root)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Preparando caso: {case_id}")
    print(f"Fonte: {brats_root}")
    print(f"Destino: {output_dir}")
    
    # Encontrar ficheiros do caso
    case_files = list(brats_root.glob(f"**/{case_id}*.nii.gz"))
    
    if not case_files:
        print(f"❌ Erro: Não encontrei ficheiros para {case_id} em {brats_root}")
        print(f"   Procurando: {brats_root}/**/{case_id}*.nii.gz")
        return False
    
    print(f"   Encontrados {len(case_files)} ficheiros")
    
    # Mapeamento de nomes BraTS para nomes PY-BRAIN
    name_mapping = {
        "_flair.nii.gz": ("flair.nii.gz", "FLAIR"),
        "_t1.nii.gz": ("t1.nii.gz", "T1"),
        "_t1ce.nii.gz": ("t1c.nii.gz", "T1c"),
        "_t2.nii.gz": ("t2.nii.gz", "T2"),
        "_seg.nii.gz": ("ground_truth.nii.gz", "Ground Truth"),
    }
    
    copied = 0
    for src_file in case_files:
        src_name = src_file.name
        
        for suffix, (dst_name, modality) in name_mapping.items():
            if suffix in src_name:
                dst_file = output_dir / dst_name
                print(f"   📄 {modality}: {src_name} → {dst_name}")
                shutil.copy2(src_file, dst_file)
                copied += 1
                break
    
    print(f"\n✅ Copiados {copied} ficheiros para {output_dir}")
    
    # Verificar ficheiros essenciais
    required = ["t1.nii.gz", "t1c.nii.gz", "t2.nii.gz", "flair.nii.gz"]
    missing = [f for f in required if not (output_dir / f).exists()]
    
    if missing:
        print(f"⚠️  Ficheiros em falta: {missing}")
        return False
    
    print("✅ Caso pronto para pipeline!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Preparar caso BraTS para PY-BRAIN")
    parser.add_argument("--case", required=True, help="ID do caso (ex: BraTS2021_00000)")
    parser.add_argument("--output", default=None, help="Diretório de saída")
    parser.add_argument("--brats-root", default=None, help="Raiz dos dados BraTS")
    
    args = parser.parse_args()
    
    # Determinar diretório de saída padrão
    if args.output is None:
        output = Path("nifti/monai_ready") / args.case.lower().replace("braTS", "brats")
    else:
        output = Path(args.output)
    
    success = prepare_brats_case(args.case, output, args.brats_root)
    
    if success:
        print(f"\nPróximo passo: criar session.json apontando para {output}")
    else:
        exit(1)

if __name__ == "__main__":
    main()
