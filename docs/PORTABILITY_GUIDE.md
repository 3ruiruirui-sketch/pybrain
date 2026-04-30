# Portability Guide — Replicating PY-BRAIN

Follow these steps to move this project to a new computer (e.g., Perplexity, a remote GPU server, or another Mac).

## 1. Prepare the Files

The project consists of three main parts:
1.  **Code**: All `.py` files and the `pybrain/` folder.
2.  **Model Weights (Large)**: The `models/brats_bundle/` folder (~1.3 GB).
3.  **Data**: Your DICOM folders.

### Step 1a: Create a Zip Archive
To move the code quickly, run this from *inside* the project folder:
```bash
zip -r py_brain_code.zip . -x "venv/*" "results/*" ".git/*" "models/*"
```

### Step 1b: Zip the Weights Separately
Run this from *inside* the project folder:
```bash
zip -r py_brain_models.zip models/brats_bundle/
```

## 2. Set Up the New Machine

Copy both `.zip` files to the new machine and unzip them into a new folder.

### Step 2a: Run the Automated Setup
I have provided a script called `portable_setup.sh` to automate the installation.
```bash
chmod +x portable_setup.sh
./portable_setup.sh
```

### Step 2b: Verify Hardware
The pipeline automatically detects your hardware:
*   **Mac (Apple Silicon)**: Uses `mps` for acceleration.
*   **Linux/Windows (NVIDIA)**: Uses `cuda` for acceleration.
*   **Other**: Falls back to `cpu`.

## 3. Running the Pipeline

Once set up, activate the environment and run the main entry point:
```bash
source venv/bin/activate
python3 run_pipeline.py
```

## Troubleshooting

### "Module Not Found"
If a module is missing, install it manually:
```bash
pip install -r requirements.txt
```

### "Dcm2niix not found"
The pipeline requires `dcm2niix` for DICOM conversion.
*   **Mac**: `brew install dcm2niix`
*   **Ubuntu/Linux**: `sudo apt-get install dcm2niix`
*   **Windows**: Download the `.exe` from the official GitHub and add it to your PATH.

---
> [!IMPORTANT]
> **Large Results**: Do NOT try to move the `results/`# Radiomics & Analysis
pyradiomics
scikit-learn
pandas
numpy>=1.24.0,<2.0.0
 It contains many large NIfTI files and can easily exceed 10GB.
