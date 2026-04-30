import torch

model_path = "/Users/ssoares/Downloads/PY-BRAIN/models/BrainIAC/segmentation.ckpt"

# Carregar no CPU primeiro para verificar
checkpoint = torch.load(model_path, map_location="cpu")

if "state_dict" in checkpoint:
    print("✅ Checkpoint Válido detetado!")
    # Ver quantas camadas o modelo tem
    num_layers = len(checkpoint["state_dict"].keys())
    print(f"🧠 O cérebro da tua IA tem {num_layers} camadas de neurónios treinadas.")
else:
    print("⚠️ O ficheiro parece estar num formato diferente ou incompleto.")
