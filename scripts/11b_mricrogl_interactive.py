#!/usr/bin/env python3
"""
Stage 11b — MRIcroGL Interactive Mass Effect Setup
==================================================
Automates the setup of MRIcroGL for clinical Mass Effect evaluation.
Launches the GUI with everything pre-configured.
"""

import sys
import subprocess
from pathlib import Path

# Setup Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.session_loader import get_session, get_paths  # type: ignore
except ImportError:
    # Minimal fallback session loading if modular loader fails
    import json
    def get_session():
        sess_file = PROJECT_ROOT / "session.json"
        if sess_file.exists():
            with open(sess_file, "r") as f: return json.load(f)
        return {}
    def get_paths(sess):
        return {
            "monai_dir": PROJECT_ROOT / "results" / sess.get("session_id", "") / "nifti" / "monai_ready",
            "output_dir": PROJECT_ROOT / "results" / sess.get("session_id", ""),
            "extra_dir": PROJECT_ROOT / "results" / sess.get("session_id", "") / "nifti" / "extra_sequences"
        }

def run():
    print("\n" + "═"*60)
    print("  PY-BRAIN — MRIcroGL MASS EFFECT AUTO-SETUP")
    print("═"*60)

    sess = get_session()
    paths = get_paths(sess)
    
    # Files
    t1c  = paths["monai_dir"] / "t1c_resampled.nii.gz"
    seg  = paths["output_dir"] / "segmentation_full.nii.gz"
    
    if not t1c.exists():
        t1c = paths["monai_dir"] / "t1.nii.gz"
    
    if not seg.exists():
        print(f"  ❌ Erro: Segmentação não encontrada em {seg}")
        print("  Executa a Stage 3 primeiro.")
        return

    # MRIcroGL Logic
    mricrogl_app = Path("/Applications/MRIcroGL.app/Contents/MacOS/MRIcroGL")
    if not mricrogl_app.exists():
        print("  ❌ MRIcroGL não encontrado em /Applications/.")
        return

    # Create the MRIcroGL Python Script (Internal script for gl module)
    setup_script = paths["output_dir"] / "mricrogl_setup_mass_effect.py"
    
    # Path conversions for MRIcroGL
    def p(path): return str(path).replace("\\", "/")

    gl_script = f"""import gl
gl.resetdefaults()
gl.loadimage('{p(t1c)}')
gl.overlayload('{p(seg)}')
gl.colorname(1, 'actc')
gl.opacity(1, 50)
gl.shadername('glass')
gl.viewrendering(1)
gl.shaderquality1to10(8)
gl.clipvisible(1)
gl.azimuthelevation(110, 15)
print('PY-BRAIN: Mass Effect Setup Complete. Use the Clip tool to explore.')
"""

    with open(setup_script, "w") as f:
        f.write(gl_script)

    print(f"  ✅ Configuração gerada: {setup_script.name}")
    print(f"  🚀 Abrindo MRIcroGL com Anatomia + Tumor...")
    
    # Launch
    subprocess.Popen([str(mricrogl_app), str(setup_script)])
    print("\n  Instruções:")
    print("  1. Usa o Rato para rodar o cérebro.")
    print("  2. A ferramenta 'Clip' no painel lateral permite fazer os cortes.")
    print("  3. Observa como a cor alaranjada (tumor) desloca as estruturas pretas (ventrículos).")
    print("═"*60 + "\n")

if __name__ == "__main__":
    run()
