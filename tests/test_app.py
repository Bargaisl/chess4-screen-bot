"""Тесты логики интерфейса: Tracker, геометрия стрелки, конфиг, smoke GUI."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chessvision.board import BLACK, WHITE, START_FEN, Position
from chessvision.app import (DEFAULT_CONFIG, REBUILD_STABILITY, Tracker,
                             cell_center, fmt_eval, load_config, save_config)
import chessvision.app as A

fails = []


def check(name, cond, extra=""):
    print(("  OK   " if cond else "  FAIL ") + name + ("" if cond else "  " + str(extra)))
    if not cond:
        fails.append(name)


START = START_FEN.split()[0]
AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"


def test_tracker_first_read():
    print("[Tracker: первое распознавание]")
    t = Tracker(WHITE)
    check("первое чтение не применяется", t.update(START) is False)
    check("второе чтение применяется", t.update(START) is True)
    check("позиция создана", t.pos is not None)
    check("ход белых", t.pos.turn == WHITE)
    check("расстановка верна", t.placement == START)
    check("тот же экран — не изменение", t.update(START) is False)
    check("seq не растёт зря", t.seq == 1, t.seq)


def test_tracker_move():
    print("[Tracker: ход с экрана]")
    t = Tracker(WHITE)
    t.update(START)
    t.update(START)
    check("ход распознан", t.update(AFTER_E4) is False)   # первое чтение
    check("ход применён", t.update(AFTER_E4) is True)
    check("теперь ход чёрных", t.pos.turn == BLACK, t.pos.turn)
    check("расстановка после хода", t.placement == AFTER_E4)
    check("ход сохранён для подсветки", t.last_move is not None)
    uci = t.pos.uci(t.last_move) if t.last_move else None
    check("ход e2e4", uci == "e2e4", uci)
    check("seq вырос", t.seq == 2, t.seq)


def test_tracker_rebuild():
    print("[Tracker: резкая смена позиции]")
    t = Tracker(WHITE)
    t.update(START)
    t.update(START)
    other = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8"
    for _ in range(REBUILD_STABILITY):
        applied = t.update(other)
    check("позиция пересобрана (после %d подтверждений)" % REBUILD_STABILITY,
          applied is True)
    check("расстановка заменена", t.placement == other, t.placement)
    check("ход стал чёрных (не белых)", t.pos.turn == BLACK, t.pos.turn)
    check("мусор игнорируется", t.update("не фен") is False)


def test_tracker_set_turn():
    print("[Tracker: смена очереди хода вручную]")
    t = Tracker(WHITE)
    t.update(START)
    t.update(START)
    check("set_turn менят очередь", t.set_turn(BLACK) is True)
    check("turn == black", t.pos.turn == BLACK)
    check("повтор — без изменений", t.set_turn(BLACK) is False)
    check("seq вырос", t.seq == 2, t.seq)


def test_tracker_rebuild_turn():
    print("[Tracker: при пересборке очередь определяется по чьём-то ходу]")
    t = Tracker(WHITE)
    t.update(START)
    t.update(START)
    # экран «прыгнул»: чёрный конь уже на f6 — из стартовой такое одним ходом
    # нельзя, но видно, что ходили чёрные, значит очередь за белыми
    jump = "rnbqkb1r/pppppppp/5n2/8/8/8/PPPPPPPP/RNBQKBNR"
    applied = False
    for _ in range(REBUILD_STABILITY):
        applied = t.update(jump)
    check("прыжок принят после подтверждений", applied is True)
    check("ходили чёрные -> очередь белых", t.pos.turn == WHITE, t.pos.turn)
    check("расстановка заменена", t.placement == jump, t.placement)


def test_tracker_manual():
    print("[Tracker: позиция задана вручную (FEN)]")
    t = Tracker(WHITE)
    t.set_manual(Position.from_fen(START_FEN))
    check("ручная позиция принята", t.manual and t.placement == START)
    seq = t.seq
    check("чтения экрана игнорируются",
          t.update(START) is False and t.update(START) is False)
    check("расстановка не сброшена", t.placement == START)
    check("seq не растёт от экрана", t.seq == seq, t.seq)
    t.reset()
    check("«Сброс» возвращает слежение",
          t.manual is False and t.pos is None)


def test_cell_center():
    print("[геометрия: клетка -> экранный центр]")
    rect = (100, 200, 800, 800)          # x, y, w, h
    # sq 0 = a1, sq 119 = h8 (0x88: rank0 = 1-я горизонталь)
    x, y, w, h = cell_center(rect, 0, "white")         # a1
    check("a1: левый нижний при взгляде от белых",
          abs(x - 150) < 1e-6 and abs(y - 950) < 1e-6, (x, y))
    check("клетка 100px", abs(w - 100) < 1e-6)
    x, y, _, _ = cell_center(rect, 119, "white")       # h8
    check("h8: правый верхний", abs(x - 850) < 1e-6 and abs(y - 250) < 1e-6,
          (x, y))
    x, y, _, _ = cell_center(rect, 0, "black")         # взгляд от чёрных
    check("black: a1 в правом верхнем",
          abs(x - 850) < 1e-6 and abs(y - 250) < 1e-6, (x, y))
    x, y, _, _ = cell_center(rect, 119, "black")       # h8 слева снизу
    check("black: h8 в левом нижнем",
          abs(x - 150) < 1e-6 and abs(y - 950) < 1e-6, (x, y))


def test_fmt_eval():
    print("[оценка]")
    check("плюс белых", fmt_eval({"score": 123}, WHITE) == "+1.23",
          fmt_eval({"score": 123}, WHITE))
    check("оценка для чёрных инвертируется", fmt_eval({"score": 123}, BLACK) == "-1.23",
          fmt_eval({"score": 123}, BLACK))
    check("мат в 2", fmt_eval({"mate": 2}, WHITE) == "+мат в 2",
          fmt_eval({"mate": 2}, WHITE))
    check("мат для чёрных", fmt_eval({"mate": -3}, WHITE) == "-мат в 3",
          fmt_eval({"mate": -3}, WHITE))


def test_config():
    print("[config.json]")
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_cfg.json")
    old = A.CONFIG_PATH
    try:
        A.CONFIG_PATH = tmp
        cfg = dict(DEFAULT_CONFIG)
        cfg["rect"] = [10, 20, 300, 300]
        cfg["think_ms"] = 2500
        save_config(cfg)
        back = load_config()
        check("rect сохранился", back["rect"] == [10, 20, 300, 300], back["rect"])
        check("think_ms сохранился", back["think_ms"] == 2500)
        check("дефолты на месте", back["orientation"] == DEFAULT_CONFIG["orientation"])
        # повреждённый файл
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write("{oops")
        back = load_config()
        check("битый json не роняет", back["rect"] is None and
              back["poll_ms"] == DEFAULT_CONFIG["poll_ms"], back)
    finally:
        A.CONFIG_PATH = old
        if os.path.exists(tmp):
            os.unlink(tmp)


def test_gui_smoke():
    print("[GUI smoke: окна создаются]")
    try:
        import tkinter as tk
    except ImportError:
        print("  SKIP tkinter")
        return
    try:
        root = tk.Tk()
    except Exception as exc:                     # noqa: BLE001
        print(f"  SKIP нет дисплея: {exc}")
        return
    app = None
    try:
        app = A.App(root, start_threads=False)
        check("панель создана", str(root.title()) in ("chess2-screen-bot", "chess4-screen-bot"),
              root.title())
        check("оверлей создан", app.ov.winfo_exists())
        check("статус виден", app.lbl_status.cget("text") != "")
        check("метка предупреждений создана", app.lbl_warn.cget("text") is not None)
        # проверяем отрисовку стрелки без потоков
        app.screen_rect = (100, 100, 400, 400)
        app.tracker.update(START)
        app.tracker.update(START)
        pos = app.tracker.pos
        m = pos.move_from_uci("e2e4")
        app.result = {"move": m, "san": "e4", "score": 30, "mate": 0,
                      "depth": 5, "nps": 1000, "pv": ["e2e4"], "book": False}
        app.result_seq = app.tracker.seq
        app._draw_overlay(True, app.result)
        items = app.ov_canvas.find_all()
        check("на оверлее есть фигуры (стрелка+клетки)", len(items) >= 3, items)
        app._refresh()               # обновление панели вручную (без mainloop)
        check("FEN вывелся", app._last_fen == app.tracker.pos.to_fen(),
              app._last_fen)
    except Exception as exc:                     # noqa: BLE001
        check("GUI smoke", False, repr(exc))
    finally:
        try:
            if app is not None:
                app.stop_evt.set()
                app.engine_stop.set()
                app.engine.close()
            root.destroy()
        except Exception:                        # noqa: BLE001
            pass


if __name__ == "__main__":
    test_tracker_first_read()
    test_tracker_move()
    test_tracker_rebuild()
    test_tracker_rebuild_turn()
    test_tracker_manual()
    test_tracker_set_turn()
    test_cell_center()
    test_fmt_eval()
    test_config()
    test_gui_smoke()
    print()
    if fails:
        print(f"FAILURES: {len(fails)} -> {fails}")
        sys.exit(1)
    print("all app tests passed")
