"""Дымовой тест: запускает настоящее приложение на N секунд и ловит ошибки.

Поднимает окна, рабочие потоки, читает реальный экран, после чего штатно
завершается. Печатает SMOKE_INFO с итоговым состоянием.

python tests/smoke_run.py [секунд]
"""
import json
import os
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROGRAM = """
import json, sys, os
sys.path.insert(0, %(root)r)
from chessvision.app import App, _enable_dpi
import tkinter as tk

_enable_dpi()
root = tk.Tk()
app = App(root)

def finish():
    pos = app.tracker.pos
    info = {
        "status": app.status,
        "rect": list(app.screen_rect) if app.screen_rect else None,
        "fen": pos.to_fen() if pos else None,
        "confidence": round(app.tracker.confidence, 3),
        "has_move": bool(app.result and app.result.get("move")),
        "overlay_items": len(app.ov_canvas.find_all()),
        "engine": app.engine.name,
    }
    print("SMOKE_INFO " + json.dumps(info, ensure_ascii=False))
    app._on_close()

root.after(int(%(sec)s * 1000), finish)
root.mainloop()
print("SMOKE_DONE")
""" % {"root": ROOT, "sec": SECONDS}


def main():
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run([sys.executable, "-c", PROGRAM], cwd=ROOT,
                          env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True, encoding="utf-8",
                          errors="replace", timeout=SECONDS + 60)
    print("returncode:", proc.returncode)
    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.stderr.strip():
        print("--- stderr ---")
        print(proc.stderr[-4000:])
    if proc.returncode != 0 or "Traceback" in (proc.stderr or "") \
            or "SMOKE_DONE" not in (proc.stdout or ""):
        print("SMOKE FAILED")
        sys.exit(1)
    print("smoke ok: приложение поднялось и штатно завершилось")


if __name__ == "__main__":
    main()
