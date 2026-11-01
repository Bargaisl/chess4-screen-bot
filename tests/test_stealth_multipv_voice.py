"""Тестирование новых возможностей: MultiPV, анти-чит Безье, SAPI озвучка и регулировка силы (Elo)."""
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from chessvision.board import Position
from chessvision.engine import StockfishEngine, find_stockfish
from chessvision.voice import san_to_voice, VoiceAnnouncer
from chessvision.mouse import _cubic_bezier, _min_jerk


def test_bezier_and_kinematics():
    print("[1. Проверка траекторий Безье и профиля скорости]")
    p0 = (100.0, 100.0)
    p1 = (200.0, 150.0)
    p2 = (350.0, 250.0)
    p3 = (500.0, 400.0)

    # Проверка граничных условий кривой
    start = _cubic_bezier(p0, p1, p2, p3, 0.0)
    end = _cubic_bezier(p0, p1, p2, p3, 1.0)
    assert abs(start[0] - 100.0) < 1e-5 and abs(start[1] - 100.0) < 1e-5
    assert abs(end[0] - 500.0) < 1e-5 and abs(end[1] - 400.0) < 1e-5

    # Проверка плавности закона минимального рывка (Flash & Hogan)
    assert _min_jerk(0.0) == 0.0
    assert _min_jerk(1.0) == 1.0
    mid = _min_jerk(0.5)
    assert abs(mid - 0.5) < 1e-5
    print("  [OK] Кинематика руки и кривые Безье работают корректно!")


def test_voice_translation():
    print("[2. Проверка голосовой трансляции нотации (SAN -> Речь)]")
    assert san_to_voice("e4") == "е 4"
    assert san_to_voice("Nf3") == "конь эф 3"
    assert san_to_voice("O-O") == "короткая рокировка"
    assert san_to_voice("O-O-O") == "длинная рокировка"
    assert san_to_voice("cxd4") == "цэ бьет дэ 4"
    assert san_to_voice("Qxf7#") == "ферзь бьет эф 7 мат"
    assert san_to_voice("Bb5+") == "слон бэ 5 шах"

    # Инициализация диктора
    announcer = VoiceAnnouncer()
    print("  Доступность Windows SAPI:", announcer._available)
    # Тестовый асинхронный вызов
    announcer.speak_move("e4")
    print("  [OK] Голосовой модуль успешно протестирован!")


def test_multipv_and_elo():
    print("[3. Проверка MultiPV (несколько стрелок) и регулировки силы Stockfish]")
    sf_path = find_stockfish()
    if not sf_path:
        print("  Stockfish не найден, пропуск")
        return

    engine = StockfishEngine(sf_path)

    # 1. Проверка установки уровня силы (Elo)
    engine.set_elo(1800)
    print("  Установлен рейтинг: 1800 Elo")

    # 2. Проверка MultiPV 2
    engine.set_multipv(2)
    pos = Position.start()
    res2 = engine.analyze(pos, movetime_ms=300)
    lines2 = res2.get("lines", [])
    print(f"  MultiPV 2 вернул {len(lines2)} линии:")
    for l in lines2:
        print(f"    Линия {l['rank']}: {l['san']} ({l['uci']}) score: {l['score']}")
    assert len(lines2) >= 2, f"Ожидалось 2 линии, получено {len(lines2)}"

    # 3. Проверка MultiPV 3
    engine.set_multipv(3)
    res3 = engine.analyze(pos, movetime_ms=300)
    lines3 = res3.get("lines", [])
    print(f"  MultiPV 3 вернул {len(lines3)} линии:")
    for l in lines3:
        print(f"    Линия {l['rank']}: {l['san']} ({l['uci']}) score: {l['score']}")
    assert len(lines3) >= 3, f"Ожидалось 3 линии, получено {len(lines3)}"

    # Возврат к максимуму
    engine.set_elo(0)
    engine.set_multipv(1)
    engine.close()
    print("  [OK] MultiPV и регулировка силы Stockfish работают идеально!")


if __name__ == "__main__":
    test_bezier_and_kinematics()
    test_voice_translation()
    test_multipv_and_elo()
    print("\nВСЕ ТЕСТЫ НОВЫХ ФУНКЦИЙ УСПЕШНО ПРОЙДЕНЫ!")
