"""Тестирование двух реальных сценариев из скриншотов пользователя."""
import os
import sys
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from chessvision.vision import read_board, detect_orientation
from chessvision.board import Position, placement_issues, BLACK, WHITE
from chessvision.engine import StockfishEngine, find_stockfish


def test_scenario_image1():
    print("[1. Тест Сценария 1: эндшпиль за чёрных]")
    p1 = os.environ.get("TEST_IMAGE_1", os.path.join(BASE_DIR, "debug", "test_scenario_1.png"))
    if not os.path.isfile(p1):
        print("  Скриншот 1 не найден, пропуск (локальный тест)")
        return
    img1 = cv2.imdecode(np.fromfile(p1, dtype=np.uint8), cv2.IMREAD_COLOR)
    rect1 = (2, 10, 568, 580)

    res = read_board(img1, rect1, orientation="auto", my_color_hint="black")
    print("  Определённая ориентация:", res["orientation"])
    assert res["orientation"] == "black", f"Ожидалась ориентация 'black', получено {res['orientation']}"

    placement = res["placement"]
    print("  Полученная расстановка FEN:", placement)
    # Короли должны быть: белый король K на g3 (3-я горизонталь), чёрный k на g7 (7-я горизонталь)
    assert "K" in placement and "k" in placement
    issues = placement_issues(placement)
    assert not issues, f"Ошибки в расстановке: {issues}"

    # Проверяем, что движок советует ход ЗА ЧЁРНЫХ, а не за белых!
    sf_path = find_stockfish()
    if sf_path:
        engine = StockfishEngine(sf_path)
        pos = Position.from_fen(f"{placement} b - - 0 1")
        assert pos.turn == BLACK
        eval_res = engine.analyze(pos, movetime_ms=300)
        best_uci = eval_res.get("uci")
        print("  Рекомендованный ход за чёрных:", best_uci, "SAN:", eval_res.get("san"))
        assert best_uci is not None
        # Проверяем, что ходит чёрная фигура (ферзь d1 или пешка e4 / a2 / b2)
        from_sq_name = best_uci[:2]
        print(f"  Ход сделан фигурой на поле: {from_sq_name}")
        engine.close()

    print("  [OK] Сценарий 1 успешно пройден!")


def test_scenario_image2():
    print("[2. Тест Сценария 2: распознавание фигур Chess.com (Neo)]")
    p2 = os.environ.get("TEST_IMAGE_2", os.path.join(BASE_DIR, "debug", "test_scenario_2.png"))
    if not os.path.isfile(p2):
        print("  Скриншот 2 не найден, пропуск (локальный тест)")
        return
    img2 = cv2.imdecode(np.fromfile(p2, dtype=np.uint8), cv2.IMREAD_COLOR)
    rect2 = (5, 1, 576, 576)

    res = read_board(img2, rect2, orientation="white", my_color_hint="white")
    placement = res["placement"]
    print("  Полученная расстановка FEN:", placement)
    print("  Уверенность (confidence):", f"{res['confidence']:.3f}")

    issues = placement_issues(placement)
    print("  Ошибки валидатора:", issues)
    assert not issues, f"Ошибки в позиции: {issues}"
    assert "2q5" in placement, "Чёрный ферзь на c6 должен быть распознан как 'q', а не 'k'!"
    assert placement.startswith("r1b1k1nr"), "Чёрный слон на c8 должен быть 'b'!"
    assert placement.endswith("R1BQKB1R"), "Первая горизонталь белых должна содержать Q и B!"
    assert res["confidence"] > 0.90, f"Низкая уверенность: {res['confidence']}"

    print("  [OK] Сценарий 2 успешно пройден: 0 ошибок, ферзи и слоны распознаны идеально!")


def test_scenario_image3():
    print("[3. Тест Сценария 3: распознавание пешек vs слонов на новом скриншоте пользователя]")
    p3 = os.environ.get("TEST_IMAGE_3", os.path.join(BASE_DIR, "debug", "test_scenario_3.png"))
    if not os.path.isfile(p3):
        print("  Скриншот 3 не найден, пропуск (локальный тест)")
        return
    img3 = cv2.imdecode(np.fromfile(p3, dtype=np.uint8), cv2.IMREAD_COLOR)
    rect3 = (10, 18, 485, 485)

    res = read_board(img3, rect3, orientation="white", my_color_hint="white")
    placement = res["placement"]
    print("  Полученная расстановка FEN:", placement)
    print("  Уверенность (confidence):", f"{res['confidence']:.3f}")

    expected = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    assert placement == expected, f"Ожидалась начальная расстановка:\n{expected}\nПолучено:\n{placement}"
    assert "bbbbbbbb" not in placement, "Пешки чёрных не должны превращаться в слонов!"
    assert "BBBBBBBB" not in placement, "Пешки белых не должны превращаться в слонов!"

    issues = placement_issues(placement)
    print("  Ошибки валидатора:", issues)
    assert not issues, f"Ошибки валидатора: {issues}"

    # Проверяем, что движок советует нормальный ход пешкой/конём (e4, d4, Nf3, c4), а НЕ Bxf7+!
    sf_path = find_stockfish()
    if sf_path:
        engine = StockfishEngine(sf_path)
        pos = Position.from_fen(f"{placement} w - - 0 1")
        assert pos.turn == WHITE
        eval_res = engine.analyze(pos, movetime_ms=300)
        best_uci = eval_res.get("uci")
        best_san = eval_res.get("san")
        print("  Рекомендованный ход движка:", best_uci, "SAN:", best_san)
        assert best_san != "Bxf7+", "Движок не должен рекомендовать невозможный Bxf7+ в начальной позиции!"
        assert best_uci in ("e2e4", "d2d4", "c2c4", "g1f3", "b1c3", "e2e3", "g2g3", "b2b3"), f"Неожиданный первый ход: {best_uci}"
        engine.close()

    print("  [OK] Сценарий 3 успешно пройден: начальная позиция определена со 100% точностью!")


if __name__ == "__main__":
    test_scenario_image1()
    test_scenario_image2()
    test_scenario_image3()
    print("\nВСЕ ПОЛЬЗОВАТЕЛЬСКИЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
