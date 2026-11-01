"""Отладка распознавания: печатает маски и оценки по клеткам."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from test_vision import render_board, placement_of
from chessvision.board import START_FEN, WHITE, BLACK
from chessvision.templates import TemplateBank, fill_silhouette, normalize_mask
from chessvision import vision as V

FEN = sys.argv[1] if len(sys.argv) > 1 else "8/8/8/4k3/8/8/8/4K3 w - - 0 1"
CELL = int(sys.argv[2]) if len(sys.argv) > 2 else 64

want = placement_of(FEN)
# разворачиваем расстановку в 8 символов на ряд
_exp_rows = []
for r in want.split("/"):
    line = []
    for ch in r:
        line.extend(["1"] * int(ch)) if ch.isdigit() else line.append(ch)
    while len(line) < 8:
        line.append("1")
    _exp_rows.append(line)
img = render_board(want, cell_px=CELL)
rect = V.detect_board(img)
print("rect:", rect, "expect:", (60, 60, 8 * CELL, 8 * CELL))

bank = TemplateBank()
res = V.read_board(img, rect, bank=bank)
print("got :", res["placement"])
print("want:", want)
print("conf:", round(res["confidence"], 3))

x, y, w, h = rect
cw, ch = w / 8.0, h / 8.0
rows = want.split("/")
print("\nper-cell:")
for row in range(8):
    line = []
    for col in range(8):
        x0, y0 = int(round(col * cw)), int(round(row * ch))
        x1, y1 = int(round((col + 1) * cw)), int(round((row + 1) * ch))
        patch = img[y + y0:y + y1, x + x0:x + x1]
        piece, score, med0, bd0 = V._classify(patch, bank, None)
        bg = V._corner_bg(patch)
        mask = V._cell_mask(patch, bg)
        from chessvision.templates import fill_silhouette as _fs
        sil = _fs(mask)
        ys, xs = np.nonzero(sil)
        med = -1.0
        if len(xs):
            reg = patch[ys, xs].astype(np.float32).reshape(-1, 3)
            med = float(np.median(reg @ np.array([0.114, 0.587, 0.299], np.float32)))
        want_ch = _exp_rows[row][col]
        got = "?"
        if piece:
            from chessvision.board import PIECE_CHARS
            color = WHITE if V._is_white(med0, V._lum(V._median_bg(patch)),
                                          bd0) else BLACK
            got = PIECE_CHARS[color | piece]
        mark = " " if got == want_ch else "!"
        line.append(f"{want_ch}->{got}{mark}({score:.2f},med={med:.0f},bg={V._lum(bg):.0f})")
    print(" ".join(line))
