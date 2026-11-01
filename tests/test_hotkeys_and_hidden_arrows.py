"""Тест горячих клавиш и режима «Без стрелок» (скрытная игра по голосу)."""
import os
import sys
import tkinter as tk

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from chessvision.app import App
from chessvision.board import WHITE, BLACK


def test_hotkeys_and_hidden_arrows():
    print("[1. Проверка инициализации GUI и элементов]")
    root = tk.Tk()
    root.withdraw()
    app = App(root, start_threads=False)

    # 1. Проверяем выбор режима "Без стрелок" через combobox и обработчик
    app.cmb_mpv.set("Без стрелок (только голос / окно)")
    app._on_multipv()
    assert app.cfg["multipv"] == 0, f"Expected multipv 0, got {app.cfg['multipv']}"
    assert app.cmb_mpv.get() == "Без стрелок (только голос / окно)"
    print("  [OK] Режим 'Без стрелок' выбран корректно (multipv == 0)!")

    # 2. Проверяем переключение очередей хода (F6 / _on_turn)
    app.btn_turn.configure(text="Белые")
    app._on_turn()
    assert app.btn_turn["text"] == "Чёрные"
    assert app.cfg["turn_hint"] == "black"
    app._on_turn()
    assert app.btn_turn["text"] == "Белые"
    assert app.cfg["turn_hint"] == "white"
    print("  [OK] Переключение хода (F6) работает в обе стороны!")

    # 3. Проверяем переключение 'Мои фигуры' (F7 / _toggle_my_color)
    app.cfg["my_color"] = "white"
    app.cmb_my.set("Белые")
    app._toggle_my_color()
    assert app.cfg["my_color"] == "black"
    assert app.cmb_my.get() == "Чёрные"
    app._toggle_my_color()
    assert app.cfg["my_color"] == "white"
    assert app.cmb_my.get() == "Белые"
    print("  [OK] Переключение стороны 'Мои фигуры' (F7) работает!")

    # 4. Проверяем Пуск / Пауза (F4 / _toggle)
    initial_enabled = app.enabled
    app._toggle()
    assert app.enabled != initial_enabled
    app._toggle()
    assert app.enabled == initial_enabled
    print("  [OK] Пуск / Пауза (F4) работает!")

    # 5. Проверяем Сброс (F5 / _reset)
    app.tracker.update("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", 0.95)
    app.tracker.update("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", 0.95)
    assert app.tracker.pos is not None
    app._reset()
    assert app.tracker.pos is None
    print("  [OK] Сброс позиции (F5) работает!")

    # 6. Проверяем оверлей при multipv == 0 (стрелки не рисуются)
    app.screen_rect = (0, 0, 400, 400)
    app.result = {
        "move": (12, 28, 0, 0),
        "uci": "e2e4",
        "san": "e4",
        "score": 35,
        "lines": [{"rank": 1, "move": (12, 28, 0, 0), "san": "e4", "score": 35}]
    }
    app.result_seq = app.tracker.seq
    app.overlay_visible = True
    app.cfg["multipv"] = 0
    app._draw_overlay(True, app.result)
    # При multipv == 0 стрелки не должны быть созданы
    items = app.ov_canvas.find_all()
    assert len(items) == 0, f"Ожидался пустой оверлей без стрелок, найдено элементов: {len(items)}"
    print("  [OK] При multipv == 0 оверлей полностью чист (0 стрелок)!")

    # 7. Проверяем переключение на 1 стрелку через комбобокс
    app.cmb_mpv.set("1 лучшая стрелка")
    app._on_multipv()
    assert app.cfg["multipv"] == 1
    app._draw_overlay(True, app.result)
    items1 = app.ov_canvas.find_all()
    assert len(items1) > 0, "Стрелка должна появиться при выборе 1 стрелки!"
    print("  [OK] При выборе '1 лучшая стрелка' стрелка появляется!")

    # 8. Проверяем переключение режима автохода (Тянуть vs 2 клика)
    init_drag = app.cfg.get("drag_drop", True)
    app._toggle_drag()
    assert app.cfg["drag_drop"] != init_drag
    assert app.btn_drag["text"] == ("2 клика" if init_drag else "Тянуть")
    app._toggle_drag()
    assert app.cfg["drag_drop"] == init_drag
    assert app.btn_drag["text"] == ("Тянуть" if init_drag else "2 клика")
    print("  [OK] Переключение режима мыши 'Тянуть / 2 клика' работает!")

    app._on_close()
    print("\nВСЕ ТЕСТЫ ГОРЯЧИХ КЛАВИШ И СКРЫТЫХ СТРЕЛОК УСПЕШНО ПРОЙДЕНЫ!")


if __name__ == "__main__":
    test_hotkeys_and_hidden_arrows()
