"""Подбор правила цвета фигуры: ядро силуэта vs цвет клетки."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2

from test_vision import render_board
from chessvision.board import CHAR_TO_PIECE, WHITE, BLACK
from chessvision import vision as V
from chessvision.templates import fill_silhouette, TemplateBank

STYLES = [
    dict(cell_px=64, light=(240, 230, 190), dark=(160, 100, 60)),
    dict(cell_px=56, light=(255, 255, 255), dark=(0, 102, 153)),
    dict(cell_px=72, light=(238, 238, 238), dark=(118, 150, 86)),
    dict(cell_px=48, light=(222, 210, 186), dark=(106, 82, 68)),
    dict(cell_px=64, light=(240, 240, 240), dark=(70, 70, 70)),
    dict(cell_px=40, light=(255, 245, 220), dark=(90, 120, 90)),
    dict(cell_px=80, light=(214, 234, 216), dark=(106, 160, 110)),
    dict(cell_px=52, light=(255, 255, 255), dark=(34, 85, 136)),
]

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
]


def expand(placement):
    out = []
    for r in placement.split("/"):
        line = []
        for ch in r:
            line.extend(["1"] * int(ch)) if ch.isdigit() else line.append(ch)
        out.append(line)
    return out


def collect():
    data = []      # (med, sq_lum, true_color, label)
    for si, style in enumerate(STYLES):
        for fen in FENS:
            want = fen.split()[0]
            img = render_board(want, **style)
            rect = V.detect_board(img)
            if rect is None:
                continue
            x, y, w, h = rect
            cw, ch = w / 8.0, h / 8.0
            exp = expand(want)
            bank = TemplateBank()
            for row in range(8):
                for col in range(8):
                    wc = exp[row][col]
                    if wc == "1":
                        continue
                    tc = WHITE if wc.isupper() else BLACK
                    x0, y0 = int(round(col * cw)), int(round(row * ch))
                    x1, y1 = int(round((col + 1) * cw)), int(round((row + 1) * ch))
                    patch = img[y + y0:y + y1, x + x0:x + x1]
                    sq = V._median_bg(patch)
                    sq_lum = V._lum(sq)
                    ptype, score, med, bd = V._classify(patch, bank, None)
                    if ptype is None:
                        continue
                    feats = {"frac_bright": 0.0, "frac_dark": 0.0, "bd": bd}
                    data.append((med, sq_lum, tc, f"s{si} {fen[:12]} {wc}",
                                 feats))
    return data


def evaluate(data):
    print(f"n={len(data)}")
    # правила
    def acc(fn):
        bad = [d for d in data if fn(d) != d[2]]
        return len(bad), bad

    rules = {
        "med > 75": lambda d: WHITE if d[0] > 75 else BLACK,
        "med > sq-40": lambda d: WHITE if d[0] > d[1] - 40 else BLACK,
        "med > sq-25": lambda d: WHITE if d[0] > d[1] - 25 else BLACK,
        "med > sq-15": lambda d: WHITE if d[0] > d[1] - 15 else BLACK,
        "med > (sq+55)/2": lambda d: WHITE if d[0] > (d[1] + 55) / 2 else BLACK,
        "med > sq*0.6": lambda d: WHITE if d[0] > d[1] * 0.6 else BLACK,
        "med > max(75, sq-30)": lambda d: WHITE if d[0] > max(75, d[1] - 30) else BLACK,
        "med > min(75, sq-15)": lambda d: WHITE if d[0] > min(75, d[1] - 15) else BLACK,
        "med > 0.5*sq+20": lambda d: WHITE if d[0] > 0.5 * d[1] + 20 else BLACK,
        "med > 0.5*sq+35": lambda d: WHITE if d[0] > 0.5 * d[1] + 35 else BLACK,
        "med > sq-12": lambda d: WHITE if d[0] > d[1] - 12 else BLACK,
        "med > sq-8": lambda d: WHITE if d[0] > d[1] - 8 else BLACK,
        "med > sq-5": lambda d: WHITE if d[0] > d[1] - 5 else BLACK,
        "med > sq-2": lambda d: WHITE if d[0] > d[1] - 2 else BLACK,
        "med > sq": lambda d: WHITE if d[0] > d[1] else BLACK,
    }
    for name, fn in rules.items():
        n, bad = acc(fn)
        print(f"  {name:24} errors={n:4d} acc={1 - n / len(data):.4f}")

    # правила на основе признака bd (доля светлых − доля тёмных в ядре)
    feats_rules = {
        "bd > -0.1": lambda d: WHITE if d[4]["bd"] > -0.1 else BLACK,
        "bd > 0": lambda d: WHITE if d[4]["bd"] > 0 else BLACK,
        "med>90 OR bd>-0.1": lambda d: WHITE if (d[0] > 90 or d[4]["bd"] > -0.1) else BLACK,
        "FINAL _is_white": lambda d: WHITE if V._is_white(d[0], d[1], d[4]["bd"]) else BLACK,
    }
    for name, fn in feats_rules.items():
        n, bad = acc(fn)
        print(f"  {name:24} errors={n:4d} acc={1 - n / len(data):.4f}")

    # лучший линейный порог: med > a*sq + b
    best = (1e9, 0, 0)
    for a10 in range(0, 11):
        a = a10 / 10
        for b in range(0, 161, 5):
            n = sum(1 for d in data if (d[0] > a * d[1] + b) != (d[2] == WHITE))
            if n < best[0]:
                best = (n, a, b)
    print(f"  best linear: med > {best[1]}*sq + {best[2]}  errors={best[0]}")

    # комбинации: WHITE если (med > t1) ИЛИ (med > a*sq + b)
    best2 = (1e9,)
    for t1 in range(60, 141, 5):
        for a10 in range(0, 11):
            a = a10 / 10
            for b in range(-40, 61, 5):
                n = 0
                for d in data:
                    got = WHITE if (d[0] > t1 or d[0] > a * d[1] + b) else BLACK
                    if got != d[2]:
                        n += 1
                if n < best2[0]:
                    best2 = (n, t1, a, b)
    print(f"  best OR: med > {best2[1]} OR med > {best2[2]}*sq + {best2[3]}  "
          f"errors={best2[0]}")

    best3 = (1e9,)
    for t1 in range(60, 141, 5):
        for a10 in range(0, 11):
            a = a10 / 10
            for b in range(-40, 61, 5):
                n = 0
                for d in data:
                    got = WHITE if (d[0] > t1 and d[0] > a * d[1] + b) else BLACK
                    if got != d[2]:
                        n += 1
                if n < best3[0]:
                    best3 = (n, t1, a, b)
    print(f"  best AND: med > {best3[1]} AND med > {best3[2]}*sq + {best3[3]}  "
          f"errors={best3[0]}")

    # min/max пороги
    best4 = (1e9,)
    for k10 in range(0, 11):
        k = k10 / 10
        for t1 in range(50, 141, 5):
            n = sum(1 for d in data
                    if (d[0] > min(t1, k * d[1])) != (d[2] == WHITE))
            if n < best4[0]:
                best4 = (n, t1, k)
    print(f"  best min: med > min({best4[1]}, {best4[2]}*sq)  errors={best4[0]}")

    best5 = (1e9,)
    for k10 in range(0, 11):
        k = k10 / 10
        for t1 in range(50, 141, 5):
            n = sum(1 for d in data
                    if (d[0] > max(t1, k * d[1])) != (d[2] == WHITE))
            if n < best5[0]:
                best5 = (n, t1, k)
    print(f"  best max: med > max({best5[1]}, {best5[2]}*sq)  errors={best5[0]}")

    # с признаком доли светлых/тёмных: WHITE если (med>t1) или
    # (med > a*sq + b И bright-dark > c)
    best6 = (1e9,)
    for t1 in range(80, 121, 5):
        for a10 in range(6, 11):
            a = a10 / 10
            for b in range(0, 21, 5):
                for c10 in range(-4, 5):
                    c = c10 / 10
                    n = 0
                    for d in data:
                        med, sq = d[0], d[1]
                        bd = d[4]["bd"]
                        got = WHITE if (med > t1 or
                                        (med > a * sq + b and bd > c)) else BLACK
                        if got != d[2]:
                            n += 1
                    if n < best6[0]:
                        best6 = (n, t1, a, b, c)
    print(f"  best w/frac: med>{best6[1]} OR (med>{best6[2]}*sq+{best6[3]} "
          f"AND bd>{best6[4]})  errors={best6[0]}")

    # статистика bd по классам в «перекрытии» med<=120
    dw = [d[4]["bd"] for d in data if d[2] == WHITE and d[0] <= 130]
    db = [d[4]["bd"] for d in data if d[2] == BLACK and d[0] <= 130]
    if dw and db:
        print(f"  overlap bd: white p10..p90={[round(float(np.percentile(dw, p)), 2) for p in (10, 50, 90)]}"
              f" black={[round(float(np.percentile(db, p)), 2) for p in (10, 50, 90)]}")

    # ошибки лучшего правила med>75
    n, bad = acc(rules["med > 75"])
    print("\nerrors of med>75:")
    for d in bad[:40]:
        print(f"   {d[3]:34} med={d[0]:6.1f} sq={d[1]:6.1f} delta={d[0] - d[1]:6.1f}")

    # распределение (med - sq) по классам
    dw = sorted(d[0] - d[1] for d in data if d[2] == WHITE)
    db = sorted(d[0] - d[1] for d in data if d[2] == BLACK)
    print("\ndelta = med - sq:")
    print("  white p5,p25,p50,p75,p95:",
          [round(float(np.percentile(dw, p)), 1) for p in (5, 25, 50, 75, 95)])
    print("  black p5,p25,p50,p75,p95:",
          [round(float(np.percentile(db, p)), 1) for p in (5, 25, 50, 75, 95)])


if __name__ == "__main__":
    evaluate(collect())
