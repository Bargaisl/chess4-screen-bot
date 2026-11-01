"""Диагностика реального экрана: что именно видит распознавание.

Снимает экран, ищет доску, читает её с подробностями по каждой клетке и
сохраняет картинки в debug/:
  debug/screen.png  — весь снимок с найденным прямоугольником;
  debug/board.png   — доска с сеткой и подписями клеток;
  debug/cells.png   — мозаика занятых клеток (патч + маска + силуэт).

python tests/diag_screen.py [rect]
  rect — необязательно "left,top,width,height" (иначе берётся config.json
  или автопоиск).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEBUG = os.path.join(ROOT, "debug")
os.makedirs(DEBUG, exist_ok=True)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from chessvision.board import PIECE_CHARS, WHITE, BLACK
from chessvision import vision as V
from chessvision.templates import default_bank, fill_silhouette, normalize_mask


def expand(placement: str):
    out = []
    for r in placement.split("/"):
        line = []
        for ch in r:
            if ch.isdigit():
                line.extend(["1"] * int(ch))
            else:
                line.append(ch)
        while len(line) < 8:
            line.append("1")
        out.append(line[:8])
    return out


def imwrite(path: str, img) -> bool:
    """cv2.imwrite молча не пишет в пути с кириллицей — кодируем сами."""
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    if not ok:
        return False
    with open(path, "wb") as fh:
        fh.write(buf.tobytes())
    return True


def main():
    rect_cfg = None
    if len(sys.argv) > 1:
        rect_cfg = [int(v) for v in sys.argv[1].split(",")]
    else:
        try:
            with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as fh:
                rect_cfg = json.load(fh).get("rect")
        except Exception:
            rect_cfg = None

    if rect_cfg:
        left, top, w, h = rect_cfg
        img = V.capture({"left": left, "top": top, "width": w, "height": h})
        rect = (0, 0, w, h)
        screen_rect = (left, top, w, h)
    else:
        img = V.capture(None)
        rect = V.detect_board(img)
        if rect is None:
            print("доска не найдена — откройте доску на экране и повторите")
            imwrite(os.path.join(DEBUG, "screen.png"), img)
            print("снимок сохранён: debug/screen.png")
            return 1
        screen_rect = rect

    print(f"прямоугольник (экран): {screen_rect}")
    x, y, w, h = rect
    crop = img[y:y + h, x:x + w].copy()
    imwrite(os.path.join(DEBUG, "board.png"), crop)

    bank = default_bank()
    res = V.read_board(img, rect)
    placement = res.get("placement")
    print(f"расстановка: {placement}")
    print(f"уверенность: {res.get('confidence'):.3f}")
    if res.get("unknown"):
        print(f"не распознано клеток: {res['unknown']}")

    # --- подробности по каждой клетке
    cw, ch = w / 8.0, h / 8.0
    board = img[y:y + h, x:x + w]
    means = np.zeros((8, 8, 3), np.float32)
    patches = {}
    for row in range(8):
        for col in range(8):
            x0, y0 = int(round(col * cw)), int(round(row * ch))
            x1, y1 = int(round((col + 1) * cw)), int(round((row + 1) * ch))
            p = board[y0:y1, x0:x1]
            patches[(col, row)] = p
            means[row, col] = p.reshape(-1, 3).mean(axis=0)
    c_hi, c_lo = V._board_colors(means.reshape(-1, 3))
    glob = (c_hi, c_lo)

    exp = expand(placement) if placement else [["1"] * 8 for _ in range(8)]
    print("\nклетка  прочтено  ожидается IoU   med    sq    bd    вердикт")
    wrong = 0
    for row in range(8):
        for col in range(8):
            p = patches[(col, row)]
            if p.size == 0 or min(p.shape[:2]) < 6:
                continue
            ptype, score, med, bd = V._classify(p, bank, glob)
            sq = V._lum(V._median_bg(p))
            if ptype is None:
                got = "."
            else:
                color = WHITE if V._is_white(med, sq, bd) else BLACK
                got = PIECE_CHARS[color | ptype]
            want = exp[row][col]
            if ptype is None and want == "1":
                continue
            ok = (got == want)
            if not ok:
                wrong += 1
            mark = "ok" if ok else "!! "
            print(f"  {chr(ord('a') + col)}{8 - row}   {got:8} {want:8} "
                  f"{score:0.3f} {med:6.1f} {sq:6.1f} {bd:+.3f}  {mark}")
    print(f"\nрасхождений с прочитанной расстановкой: {wrong}")

    # --- мозаика клеток для визуальной проверки
    tiles = []
    for row in range(8):
        for col in range(8):
            p = patches[(col, row)]
            if p.size == 0:
                continue
            bg = V._median_bg(p)
            mask = V._cell_mask(p, bg)
            sil = fill_silhouette(mask)
            strip = np.concatenate([
                cv2.resize(p, (64, 64)),
                cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (64, 64)),
                cv2.resize(cv2.cvtColor(sil, cv2.COLOR_GRAY2BGR), (64, 64)),
            ], axis=1)
            cv2.putText(strip, f"{chr(ord('a') + col)}{8 - row}", (2, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            tiles.append(strip)
    if tiles:
        cols = 8
        rows_n = (len(tiles) + cols - 1) // cols
        blank = np.zeros_like(tiles[0])
        while len(tiles) < rows_n * cols:
            tiles.append(blank)
        mosaic = np.concatenate([np.concatenate(tiles[i * cols:(i + 1) * cols],
                                                axis=1)
                                 for i in range(rows_n)], axis=0)
        imwrite(os.path.join(DEBUG, "cells.png"), mosaic)
        print("мозаика клеток (патч | маска | силуэт): debug/cells.png")
    print("доска с сеткой: debug/board.png, весь экран: debug/screen.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
