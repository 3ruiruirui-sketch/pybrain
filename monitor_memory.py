#!/usr/bin/env python3
"""
Simple memory monitoring script for PY-BRAIN inference
"""
import psutil
import time
import torch
from datetime import datetime

def monitor_memory(interval=1.0):
    """Monitor system and GPU memory during inference"""
    try:
        while True:
            # System memory
            memory = psutil.virtual_memory()
            
            # PyTorch MPS info (if available)
            mps_info = ""
            if torch.backends.mps.is_available():
                try:
                    # Try to get some MPS info if available
                    mps_info = " | MPS: Available"
                except:
                    mps_info = " | MPS: Available"
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] RAM: {memory.used/1e9:.2f}GB/{memory.total/1e9:.2f}GB ({memory.percent:.1f}%){mps_info}")
            
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    print("Starting memory monitoring... Press Ctrl+C to stop")
    monitor_memory()
