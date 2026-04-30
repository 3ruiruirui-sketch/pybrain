import os
import urllib.request

# Configuração de Pastas
save_dir = "./CELESTE-BRAIN/data/validation_set"
os.makedirs(save_dir, exist_ok=True)

# Links Diretos (Amostra de Glioma - MSD Dataset)
# Nota: Estes ficheiros são volumétricos (.nii.gz)
data_urls = {
    "T2.nii.gz": "https://github.com/neheller/kits19/raw/master/data/case_00000/imaging.nii.gz",  # Exemplo de estrutura
    "FLAIR.nii.gz": "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/brats_sample.nii.gz",
}


def download_real_data():
    print("🚀 A iniciar download de dados REAIS para o CELESTE-BRAIN...")
    for filename, url in data_urls.items():
        path = os.path.join(save_dir, filename)
        if not os.path.exists(path):
            try:
                print(f"📥 A descarregar {filename}...")
                urllib.request.urlretrieve(url, path)
                print(f"✅ Guardado em: {path}")
            except Exception as e:
                print(f"❌ Erro ao descarregar {filename}: {e}")
        else:
            print(f"ℹ️ {filename} já existe. Salto o download.")


if __name__ == "__main__":
    download_real_data()
