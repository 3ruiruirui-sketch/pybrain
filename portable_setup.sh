#!/bin/bash
# PY-BRAIN Portability Support Script
# ===================================
# This script sets up a fresh virtual environment and installs
# all medical imaging dependencies needed for the pipeline.

set -e

echo "--------------------------------------------------------"
echo "  🚀 PY-BRAIN — Replicating Environment...             "
echo "--------------------------------------------------------"

# 1. Create Virtual Environment
if [ -d "venv" ]; then
    echo "  ℹ  'venv' already exists. Re-using..."
else
    echo "  📂 Creating virtual environment..."
    python3 -m venv venv
fi

# 2. Activate Environment
source venv/bin/activate

# 3. Upgrade Pip
echo "  🐍 Upgrading pip..."
pip install --upgrade pip

# 4. Install Dependencies
echo "  📦 Installing medical imaging stack from requirements.txt..."
pip install -r requirements.txt

# 5. Build/Verify Components
echo "  🧪 Verifying installation..."

python3 <<EOF
import torch
print(f"    - Torch Version: {torch.__version__}")
if torch.backends.mps.is_available():
    print("    - 🍏 Hardware: Mac (MPS) detected")
elif torch.cuda.is_available():
    print("    - 🔥 Hardware: NVIDIA (CUDA) detected")
else:
    print("    - 🐌 Hardware: CPU only detected")

try:
    import monai
    print(f"    - MONAI Version: {monai.__version__}")
except ImportError:
    print("    - ❌ MONAI missing")

try:
    import pydicom
    print("    - 📄 Pydicom ready")
except ImportError:
    print("    - ❌ Pydicom missing")
EOF

echo "--------------------------------------------------------"
echo "  ✅ Setup Complete!                                   "
echo "  To start the pipeline, run:                          "
echo "  source venv/bin/activate && python3 run_pipeline.py  "
echo "--------------------------------------------------------"
