#!/usr/bin/env python3
"""
Utilitário de diagnóstico: corre um script Python com:
  - Contador de tempo ao vivo (heartbeat cada N segundos)
  - Timeout configurável (mata o processo se ultrapassar)
  - Exit code e duração no final

Uso:
  python3 scripts/tools/run_with_timeout.py <script.py> [timeout_segundos] [heartbeat_segundos]

Exemplos:
  python3 scripts/tools/run_with_timeout.py scripts/3_brain_tumor_analysis.py 3600 10
  python3 scripts/tools/run_with_timeout.py scripts/7_train_nnunet.py 7200 30
"""

import sys
import subprocess
import threading
import time
import os
import signal
from pathlib import Path


def run_with_heartbeat(script_path: str, timeout: int = 600, heartbeat: int = 10):
    script = Path(script_path)
    if not script.exists():
        print(f"❌ Script não encontrado: {script_path}")
        sys.exit(1)

    print(f"▶  {script.name}  |  timeout={timeout}s  |  heartbeat={heartbeat}s")
    print(f"   {'-' * 60}")

    start = time.time()
    last_output_time = [start]  # mutable ref for thread
    timed_out = [False]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        [sys.executable, "-u", str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=str(script.parent.parent.parent if "scripts" in str(script) else Path.cwd()),
    )

    output_lines: list = []

    def _reader():
        for line in proc.stdout:  # type: ignore[union-attr]
            last_output_time[0] = time.time()
            output_lines.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    def _heartbeat():
        while proc.poll() is None:
            time.sleep(heartbeat)
            if proc.poll() is not None:
                break
            elapsed = time.time() - start
            silent_for = time.time() - last_output_time[0]
            # Only print heartbeat if no output for more than heartbeat interval
            if silent_for >= heartbeat * 0.9:
                print(f"   ⏱  {elapsed:.0f}s decorridos | sem output há {silent_for:.0f}s", flush=True)
            if elapsed > timeout:
                timed_out[0] = True
                print(f"\n   ⚠️  TIMEOUT após {timeout}s — a matar processo...", flush=True)
                try:
                    proc.send_signal(signal.SIGTERM)
                    time.sleep(3)
                    if proc.poll() is None:
                        proc.kill()
                except Exception:
                    pass
                break

    hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    hb_thread.start()

    proc.wait()
    reader_thread.join(timeout=5)
    elapsed_total = time.time() - start

    print(f"   {'-' * 60}")
    if timed_out[0]:
        print(f"❌  TIMEOUT  |  {elapsed_total:.1f}s  |  exit={proc.returncode}")
        sys.exit(124)
    elif proc.returncode == 0:
        print(f"✅  OK  |  {elapsed_total:.1f}s  |  exit=0")
    else:
        print(f"❌  FALHOU  |  {elapsed_total:.1f}s  |  exit={proc.returncode}")
        sys.exit(proc.returncode)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    script_arg = sys.argv[1]
    timeout_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    hb_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    run_with_heartbeat(script_arg, timeout_arg, hb_arg)
