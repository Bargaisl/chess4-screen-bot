"""Сквозной тест: поток захвата -> распознавание -> Tracker -> движок -> оверлей.

Вместо реального экрана подставляется синтетическая картинка доски.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from chessvision.board import START_FEN, WHITE, BLACK
from chessvision import app as A

fails = []


def check(name, cond, extra=""):
    print(("  OK   " if cond else "  FAIL ") + name + ("" if cond else "  " + str(extra)))
    if not cond:
        fails.append(name)


def make_screen():
    from test_vision import render_board
    board = render_board(START_FEN.split()[0], cell_px=64)   # BGR
    h, w = board.shape[:2]
    screen = np.full((800, 1200, 3), 46, np.uint8)
    y0, x0 = 60, 300
    screen[y0:y0 + h, x0:x0 + w] = board
    # у render_board поля margin=60 вокруг доски 512x512
    expect = (x0 + 60, y0 + 60, w - 120, h - 120)
    return screen, expect


def test_live():
    print("[сквозной: захват -> распознавание -> движок -> стрелка]")
    screen, expect = make_screen()
    real_capture = A.capture
    A.capture = lambda region=None: screen.copy()
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as exc:                      # noqa: BLE001
        print(f"  SKIP нет дисплея: {exc}")
        A.capture = real_capture
        return
    app = None
    try:
        A.DEFAULT_CONFIG["think_ms"] = 700
        app = A.App(root, start_threads=False)
        app.cfg["think_ms"] = 700
        app.cfg["rect"] = None
        app.cfg["orientation"] = "white"
        app.enabled = True
        # запускаем рабочие потоки вручную (как в App, но после подмены capture)
        import threading
        threading.Thread(target=app._capture_loop, daemon=True).start()
        threading.Thread(target=app._engine_loop, daemon=True).start()
        app.wake.set()

        deadline = time.time() + 45
        while time.time() < deadline:
            root.update()
            time.sleep(0.05)
            if (app.tracker.pos is not None and app.result is not None
                    and app.result.get("move") is not None):
                break

        check("доска найдена на экране", app.screen_rect is not None,
              app.screen_rect)
        if app.screen_rect:
            got = app.screen_rect
            check("прямоугольник совпадает с доской",
                  all(abs(a - b) <= 3 for a, b in zip(got, expect)),
                  f"got {got} expect {expect}")
        check("позиция распознана", app.tracker.pos is not None)
        if app.tracker.pos is not None:
            check("расстановка верна",
                  app.tracker.placement == START_FEN.split()[0],
                  app.tracker.placement)
            check("ход белых", app.tracker.pos.turn == WHITE, app.tracker.pos.turn)
        check("движок дал ход", app.result is not None
              and app.result.get("move") is not None, app.result)
        if app.result:
            uci = app.result.get("uci")
            check("ход из стартовой позиции корректен",
                  uci in ("e2e4", "d2d4", "g1f3", "b1c3", "c2c4", "e2e3", "d2d3"),
                  uci)
            check("оценка отформатирована", "+" in A.fmt_eval(
                app.result, app.tracker.pos.turn), A.fmt_eval(app.result,
                                                              app.tracker.pos.turn))
        # оверлей должен содержать стрелку
        root.update()
        app._draw_overlay(True, app.result)
        items = app.ov_canvas.find_all()
        check("на оверлее нарисована стрелка", len(items) >= 3, len(items))
    except Exception as exc:                       # noqa: BLE001
        check("сквозной тест", False, repr(exc))
    finally:
        A.capture = real_capture
        try:
            if app is not None:
                app.stop_evt.set()
                app.engine_stop.set()
                app.wake.set()
                app.engine.close()
            root.destroy()
        except Exception:                          # noqa: BLE001
            pass


if __name__ == "__main__":
    test_live()
    print()
    if fails:
        print(f"FAILURES: {len(fails)} -> {fails}")
        sys.exit(1)
    print("live test passed")
