"""Тестирование игры за чёрных, авто-ориентации и устойчивости Stockfish."""
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from chessvision.board import Position, WHITE, BLACK
from chessvision.vision import square_to_cell, _flip_placement, detect_orientation
from chessvision.engine import StockfishEngine, find_stockfish
from chessvision.app import cell_center


def test_black_orientation_and_moves():
    print("[1. Проверка ориентации за чёрных]")
    # Экран, когда игрок играет за чёрных (чёрные снизу):
    # Верх (ряд 0): белые фигуры
    # Верх (ряд 1): белые пешки
    # Низ (ряд 6): чёрные пешки
    # Низ (ряд 7): чёрные фигуры
    screen_rows = [
        "RNBKQBNR",
        "PPPPPPPP",
        "8",
        "8",
        "8",
        "8",
        "pppppppp",
        "rnbkqbnr"
    ]
    detected = detect_orientation(screen_rows, my_color_hint="black")
    print("  Обнаруженная ориентация:", detected)
    assert detected == "black", f"Ожидалась 'black', получено {detected}"

    raw_placement = "/".join(screen_rows)
    fen_placement = _flip_placement(raw_placement)
    print("  FEN после переворота:", fen_placement)
    assert fen_placement == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

    # Теперь проверяем координаты клика автохода
    # Прямоугольник доски на экране: (100, 100, 800, 800) -> клетка 100x100
    rect = (100, 100, 800, 800)

    pos = Position.start()
    pos.turn = BLACK

    # Чёрные делают ход пешкой c7 на c5
    m_c7c5 = pos.move_from_uci("c7c5")
    assert m_c7c5 is not None
    from_sq, to_sq = m_c7c5[0], m_c7c5[1]

    # Координаты кликов автохода в ориентации 'black':
    fx, fy, _, _ = cell_center(rect, from_sq, "black")
    tx, ty, _, _ = cell_center(rect, to_sq, "black")

    print(f"  Клик c7: ({fx:.1f}, {fy:.1f}) -> клик c5: ({tx:.1f}, {ty:.1f})")

    # В ориентации 'black':
    # Ряд 6 (предпоследний снизу) — это c7 (fy ~ 750)
    # Ряд 4 (середина) — это c5 (ty ~ 550)
    # Ход идёт снизу вверх к сопернику: ty < fy!
    assert fy > ty, f"Ход чёрных должен идти снизу вверх: fy={fy}, ty={ty}"
    # Колонка c: в ориентации 'black' h=0, g=1, f=2, e=3, d=4, c=5 -> fx ~ 100 + 5.5*100 = 650
    assert abs(fx - 650) < 5
    assert abs(tx - 650) < 5
    print("  [OK] Координаты автохода за чёрных вычислены абсолютно верно!")


def test_stockfish_resilience():
    print("[2. Проверка живучести движка Stockfish]")
    sf_path = find_stockfish()
    if not sf_path:
        print("  Stockfish не найден, пропуск теста")
        return

    print("  Найден Stockfish:", sf_path)
    engine = StockfishEngine(sf_path)

    # 1. Обычный анализ
    pos = Position.start()
    res1 = engine.analyze(pos, movetime_ms=300)
    print("  Стартовый ход движка:", res1.get("uci"), "score:", res1.get("score"))
    assert res1.get("move") is not None, "Движок не выдал ход на стартовой позиции"

    # 2. Позиция с ходом чёрных
    pos.turn = BLACK
    pos.board[0x34] = pos.board[0x14]  # e2->e4
    pos.board[0x14] = 0
    pos.compute_key()
    res2 = engine.analyze(pos, movetime_ms=300)
    print("  Ответный ход чёрных:", res2.get("uci"), "score:", res2.get("score"))
    assert res2.get("move") is not None, "Движок не выдал ход за чёрных"

    # 3. Имитация сбоя: принудительно убиваем процесс Stockfish!
    print("  Тест самовосстановления: убиваем процесс Stockfish...")
    try:
        engine.proc.kill()
        engine.proc.wait(timeout=1)
    except Exception:
        pass
    print("  Процесс убит. Статус poll():", engine.proc.poll())
    assert engine.proc.poll() is not None

    # Следующий вызов analyze ДОЛЖЕН автоматически поднять Stockfish и выдать ход без [Errno 22]!
    res3 = engine.analyze(pos, movetime_ms=300)
    print("  Ход после убийства процесса:", res3.get("uci"), "score:", res3.get("score"))
    assert res3.get("move") is not None, "Движок не восстановился после гибели процесса!"
    print("  [OK] Движок успешно самовосстановился и не выбросил [Errno 22]!")

    engine.close()


if __name__ == "__main__":
    test_black_orientation_and_moves()
    test_stockfish_resilience()
    print("\nВСЕ ТЕСТЫ ЗАВЕРШЕНЫ УСПЕШНО!")
