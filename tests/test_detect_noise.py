"""Регрессия: доска под панелью браузера не должна путаться с панелью.

Реальный случай: detect_board выдал прямоугольник, начинавшийся с тулбара
(строка вкладок, адресная строка, белая страница), поэтому внутри кропа
оказались панель и половина доски — и распознавание выдавало мусор.

Экран здесь собран искусственно: сверху тёмная панель с горизонтальными
линиями и белая страница, снизу — настоящая доска.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from chessvision.vision import detect_board, _alternation_score
from test_vision import check, fails, render_board, placement_of
from chessvision.board import START_FEN


def make_browser_screen():
    """Панель браузера + белая страница + доска ниже."""
    img = np.full((1080, 1920, 3), 46, np.uint8)
    # --- тёмная панель: вкладки, адресная строка, закладки (много линий)
    img[0:200] = (44, 44, 46)
    for y in (8, 34, 60, 86, 112, 138, 164, 190):
        img[y:y + 2] = (70, 70, 74)
    for y in (20, 46, 72, 98, 124, 150, 176):
        img[y:y + 1, 100:900] = (96, 96, 100)
    # --- белая страница с «текстом»
    img[200:380] = (250, 250, 250)
    for i in range(8):
        y = 214 + i * 22
        img[y:y + 8, 320:1500] = (120, 120, 124)
    # --- настоящая доска под страницей
    board = render_board(placement_of(START_FEN), cell_px=70)
    bh, bw = board.shape[:2]
    x0, y0 = 620, 390
    img[y0:y0 + bh, x0:x0 + bw] = board
    expect = (x0 + 60, y0 + 60, bw - 120, bh - 120)   # margin=60 у render_board
    return img, expect


def test_detect_below_browser_ui():
    print("[поиск доски: панель браузера + страница + доска]")
    img, expect = make_browser_screen()
    rect = detect_board(img)
    check("доска найдена", rect is not None, rect)
    if rect is None:
        return
    ok = all(abs(a - b) <= 4 for a, b in zip(rect, expect))
    check("прямоугольник = доска, а не панель", ok, f"got {rect} expect {expect}")
    check("верх кропа не в тулбаре (y >= 400)", rect[1] >= 400, rect)


def test_toolbar_hypothesis_rejected():
    print("[гипотеза «панель браузера» отвергается чётностью]")
    img, expect = make_browser_screen()
    sx = sy = 70.0
    # гипотеза, привязанная к панели: шаг настоящей клетки, но начало в тулбаре
    bad = _alternation_score(img, 620, 8, sx, sy)
    good = _alternation_score(img, expect[0], expect[1], sx, sy)
    check("панель получает -inf", bad <= -1e8, bad)
    check("настоящая доска получает счёт", good > -1e8, good)


if __name__ == "__main__":
    test_detect_below_browser_ui()
    test_toolbar_hypothesis_rejected()
    print()
    if fails:
        print(f"FAILURES: {len(fails)} -> {fails}")
        sys.exit(1)
    print("all detect-noise tests passed")
