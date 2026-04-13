#!/usr/bin/env python3
"""
Benchmark SwinUNETR: MPS vs CPU (v7.2.4)
"""
import time
import sys
import importlib.util
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def load_stage3():
    """Import scripts/3_brain_tumor_analysis.py dynamically."""
    script_path = PROJECT_ROOT / "scripts" / "3_brain_tumor_analysis.py"
    spec = importlib.util.spec_from_file_location("stage3_module", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main

def run_benchmark():
    print("═" * 60)
    print("  SWINUNETR MPS BENCHMARK (v7.2.4)")
    print("═" * 60)
    
    try:
        stage3_main = load_stage3()
    except Exception as e:
        print(f"\n❌ Failed to load Stage 3: {e}")
        return

    t0 = time.time()
    try:
        stage3_main()
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        return

    elapsed = time.time() - t0
    mins = elapsed / 60
    
    print("\n" + "═" * 60)
    print(f"  BENCHMARK COMPLETE")
    print(f"  Total Stage 3 Time : {elapsed:.1f}s ({mins:.1f} min)")
    print(f"  Target Goal        : < 10 min (7-8 min expected)")
    print("═" * 60)

if __name__ == "__main__":
    run_benchmark()
