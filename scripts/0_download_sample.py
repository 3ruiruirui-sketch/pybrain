#!/usr/bin/env python3
import requests
from pathlib import Path
from tqdm import tqdm

# Configuração de Pastas do CELESTE-BRAIN
base_path = Path("./CELESTE-BRAIN/data/validation_set/")
base_path.mkdir(parents=True, exist_ok=True)

# URLs de exemplo (Ficheiros NIfTI reais de um caso BraTS 2018/2020)
# Nota: BraTS data URLs directas e estáveis são difíceis de manter sem autenticação.
# Usamos aqui links de repositórios públicos que hospedam amostras para testes.
files = {
    "T2.nii.gz": "https://github.com/akhanss/BraTS-2020/raw/master/data/brats20/TrainingData/BraTS20_Training_001/BraTS20_Training_001_t2.nii.gz",
    "FLAIR.nii.gz": "https://github.com/akhanss/BraTS-2020/raw/master/data/brats20/TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii.gz",
    "GT_SEG.nii.gz": "https://github.com/akhanss/BraTS-2020/raw/master/data/brats20/TrainingData/BraTS20_Training_001/BraTS20_Training_001_seg.nii.gz",
}


def download_file(url, target_path):
    """Download a file with a progress bar."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        t = tqdm(total=total_size, unit="iB", unit_scale=True, desc=target_path.name)
        with open(target_path, "wb") as f:
            for data in response.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()
        return True
    else:
        print(f"  ❌ Erro ao descarregar {target_path.name}: {response.status_code}")
        return False


def download_sample():
    print(f"\n📂 Preparando dados em: {base_path.resolve()}\n")
    for name, url in files.items():
        dest = base_path / name
        if dest.exists():
            print(f"  ✅ {name} já existe. Ignorando.")
            continue

        print(f"📥 A descarregar {name}...")
        try:
            success = download_file(url, dest)
            if success:
                print(f"  ✅ Download concluído: {name}")
        except Exception as e:
            print(f"  ❌ Falha no download de {name}: {e}")

    print("\n✅ Processo de download finalizado.")
    print("ℹ️  Nota: Se os downloads falharem (404), os links podem ter mudado.")
    print("   Nesse caso, podes colocar manualmente ficheiros .nii.gz na pasta acima.")


if __name__ == "__main__":
    download_sample()
