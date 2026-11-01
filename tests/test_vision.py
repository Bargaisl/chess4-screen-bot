"""Синтетические доски: рисуем картинку -> detect_board -> read_board == FEN."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from chessvision.board import (BISHOP, CHAR_TO_PIECE, KING, KNIGHT, PAWN,
                               QUEEN, ROOK, WHITE, BLACK, COLOR_MASK, Position,
                               START_FEN)
from chessvision.templates import GLYPHS, _candidate_fonts, render_glyph
from chessvision.vision import detect_board, read_board, _flip_placement

fails = []


def check(name, cond, extra=""):
    print(("  OK   " if cond else "  FAIL ") + name + ("" if cond else "  " + str(extra)))
    if not cond:
        fails.append(name)


def pick_font(px):
    for path in _candidate_fonts(20):
        try:
            font = ImageFont.truetype(path, px)
        except Exception:
            continue
        mask = np.array(font.getmask("\u265A"))
        if mask.size and (mask > 32).sum() > 40:
            return font
    raise RuntimeError("нет шрифта с шахматными глифами")


def render_board(placement: str, cell_px: int = 64, margin: int = 60,
                 light=(240, 230, 190), dark=(160, 100, 60),
                 piece_bg_dark=None, highlight=(), orient="white",
                 font_scale=0.82, padding=0.06) -> np.ndarray:
    """Рисует доску PIL'ом. orient='black' — переворачиваем изображение."""
    n = 8
    board_px = cell_px * n
    img = Image.new("RGB", (board_px + 2 * margin, board_px + 2 * margin),
                    (60, 60, 64))
    d = ImageDraw.Draw(img)
    rows = []
    for r in placement.split("/"):
        line = []
        for ch in r:
            if ch.isdigit():
                line.extend(["1"] * int(ch))
            else:
                line.append(ch)
        while len(line) < 8:
            line.append("1")
        rows.append(line[:8])
    font = pick_font(int(cell_px * font_scale))
    for row in range(8):
        for col in range(8):
            x0 = margin + col * cell_px
            y0 = margin + row * cell_px
            color = light if (row + col) % 2 == 0 else dark
            if (col, row) in highlight:
                color = (246, 246, 105)
            d.rectangle([x0, y0, x0 + cell_px, y0 + cell_px], fill=color)
            ch = rows[row][col]
            if ch.isdigit():
                continue
            piece = CHAR_TO_PIECE[ch.upper()] & 7
            color_piece = (250, 250, 250) if ch.isupper() else (25, 25, 30)
            outline = (0, 0, 0) if ch.isupper() else (245, 245, 245)
            gl = GLYPHS[(WHITE if ch.isupper() else BLACK, piece)]
            d.text((x0 + cell_px // 2, y0 + cell_px // 2), gl, font=font,
                   fill=color_piece, anchor="mm", stroke_width=1,
                   stroke_fill=outline)
    arr = np.array(img)
    if orient == "black":
        arr = np.rot90(arr, 2).copy()
    return np.ascontiguousarray(arr[:, :, ::-1])   # -> BGR, как с mss


def placement_of(fen):
    return fen.split()[0]


def test_detect_simple():
    print("[поиск доски: простая картинка]")
    img = render_board(placement_of(START_FEN))
    rect = detect_board(img)
    check("найдена сетка 8x8", rect is not None, rect)
    if rect:
        x, y, w, h = rect
        # доска в картинке начинается на margin=60 и размер board_px
        cw = w / 8
        ch = h / 8
        check("клетка ~64px", abs(cw - 64) < 8 and abs(ch - 64) < 8, rect)
        check("центр доски совпадает", abs((x + w / 2) - (img.shape[1] / 2)) < 12,
              (x, w, img.shape[1]))


def test_read_start():
    print("[распознавание: стартовая позиция]")
    img = render_board(placement_of(START_FEN))
    rect = detect_board(img)
    res = read_board(img, rect)
    check(f"расстановка верна ({res['confidence']:.2f})",
          res["placement"] == placement_of(START_FEN),
          f"got {res['placement']}")
    print(f"       confidence={res['confidence']:.3f} unknown={res['unknown']}")


def test_read_many():
    print("[распознавание: разные позиции/стили]")
    cases = [
        ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
         dict(cell_px=56, light=(255, 255, 255), dark=(0, 102, 153))),
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
         dict(cell_px=72, light=(238, 238, 238), dark=(118, 150, 86))),
        ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
         dict(cell_px=48, light=(222, 210, 186), dark=(106, 82, 68),
              highlight=[(4, 6), (4, 4)])),
        ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
         dict(cell_px=64, light=(240, 240, 240), dark=(70, 70, 70))),
    ]
    for fen, style in cases:
        want = placement_of(fen)
        img = render_board(want, **style)
        rect = detect_board(img)
        if rect is None:
            check(f"сетка найдена {fen[:30]}", False, rect)
            continue
        res = read_board(img, rect)
        ok = res["placement"] == want
        check(f"позиция {fen[:34]}", ok,
              f"got {res['placement']} want {want} conf={res['confidence']:.2f}")


def test_orientation():
    print("[взгляд от чёрных: клетки развёрнуты, фигуры стоят прямо]")
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    want = placement_of(fen)
    flipped = _flip_placement(want)
    # так доску видит игрок за чёрных: сверху 1-я горизонталь, справа файл a
    img = render_board(flipped)
    rect = detect_board(img)
    if rect is None:
        check("сетка найдена (чёрные)", False, rect)
        return
    res = read_board(img, rect, orientation="black")
    check(f"с flip получаем исходную расстановку ({res['placement']})",
          res["placement"] == want, f"want {want}")
    res2 = read_board(img, rect, orientation="white")
    check("без flip — зеркальная расстановка",
          res2["placement"] == flipped, res2["placement"])


def test_occupied_empty():
    print("[пустые клетки не опознаются как фигуры]")
    fen = "8/8/8/4k3/8/8/8/4K3 w - - 0 1"
    img = render_board(placement_of(fen), cell_px=64)
    rect = detect_board(img)
    res = read_board(img, rect)
    check("ровно 2 фигуры", res["placement"] == placement_of(fen),
          res["placement"])


def test_flip():
    print("[_flip_placement]")
    p = placement_of(START_FEN)
    check("flip(flip(x)) == x", _flip_placement(_flip_placement(p)) == p)
    q = "4k3/8/8/8/8/8/4P3/4K3"
    check("flip", _flip_placement(q) == "3K4/3P4/8/8/8/8/8/3k4",
          _flip_placement(q))


if __name__ == "__main__":
    test_flip()
    test_detect_simple()
    test_read_start()
    test_read_many()
    test_orientation()
    test_occupied_empty()
    print()
    if fails:
        print(f"FAILURES: {len(fails)} -> {fails}")
        sys.exit(1)
    print("all vision tests passed")
