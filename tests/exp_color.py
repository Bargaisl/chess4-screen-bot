"""Эксперимент: какой признак лучше отличает белую фигуру от чёрной."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2

from test_vision import render_board
from chessvision.board import CHAR_TO_PIECE, WHITE, BLACK, PIECE_CHARS
from chessvision import vision as V
from chessvision.templates import fill_silhouette

STYLES = [
    dict(cell_px=64, light=(240, 230, 190), dark=(160, 100, 60)),
    dict(cell_px=56, light=(255, 255, 255), dark=(0, 102, 153)),
    dict(cell_px=72, light=(238, 238, 238), dark=(118, 150, 86)),
    dict(cell_px=48, light=(222, 210, 186), dark=(106, 82, 68)),
    dict(cell_px=64, light=(240, 240, 240), dark=(70, 70, 70)),
    dict(cell_px=40, light=(255, 245, 220), dark=(90, 120, 90)),
]

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
]

METHODS = {}


def method(name):
    def deco(fn):
        METHODS[name] = fn
        return fn
    return deco


def expand(placement):
    out = []
    for r in placement.split("/"):
        line = []
        for ch in r:
            line.extend(["1"] * int(ch)) if ch.isdigit() else line.append(ch)
        out.append(line)
    return out


@method("median")
def f_median(patch, sil, bg_lum, light_lum, dark_lum):
    return V._sil_lum(patch, sil)


@method("median_core2")
def f_core(patch, sil, bg_lum, light_lum, dark_lum):
    core = cv2.erode(sil, np.ones((5, 5), np.uint8))
    if (core > 0).sum() < 20:
        core = sil
    return V._sil_lum(patch, core)


@method("median_core3")
def f_core3(patch, sil, bg_lum, light_lum, dark_lum):
    core = cv2.erode(sil, np.ones((7, 7), np.uint8))
    if (core > 0).sum() < 20:
        core = sil
    return V._sil_lum(patch, core)


@method("mode16")
def f_mode(patch, sil, bg_lum, light_lum, dark_lum):
    ys, xs = np.nonzero(sil)
    reg = patch[ys, xs].astype(np.float32)
    lum = reg @ np.array([0.114, 0.587, 0.299], np.float32)
    hist, edges = np.histogram(lum, bins=32, range=(0, 256))
    i = int(np.argmax(hist))
    return float((edges[i] + edges[i + 1]) / 2)


@method("bright_vs_dark_frac")
def f_frac(patch, sil, bg_lum, light_lum, dark_lum):
    ys, xs = np.nonzero(sil)
    reg = patch[ys, xs].astype(np.float32)
    lum = reg @ np.array([0.114, 0.587, 0.299], np.float32)
    b = float((lum > 200).mean())
    d = float((lum < 60).mean())
    return 255.0 if b > d else 0.0


def run():
    rows = {name: [] for name in METHODS}
    all_vals = {name: ([], []) for name in METHODS}   # name -> (белые, чёрные)
    adapt_rows = []          # core2 + адаптивный порог (k-means по 1D)
    err_details = []
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
            img_vals = []     # (значение, истинный цвет) для адаптивного порога
            for row in range(8):
                for col in range(8):
                    ch_ = exp[row][col]
                    if ch_ == "1":
                        continue
                    piece = CHAR_TO_PIECE[ch_.upper()]
                    true_color = WHITE if ch_.isupper() else BLACK
                    x0, y0 = int(round(col * cw)), int(round(row * ch))
                    x1, y1 = int(round((col + 1) * cw)), int(round((row + 1) * ch))
                    patch = img[y + y0:y + y1, x + x0:x + x1]
                    bg = V._corner_bg(patch)
                    mask = V._cell_mask(patch, bg)
                    ink = float((mask > 0).mean())
                    if ink > 0.72 or ink < 0.005:
                        bg = V._median_bg(patch)
                        mask = V._cell_mask(patch, bg)
                        ink = float((mask > 0).mean())
                        if ink > 0.72 or ink < 0.005:
                            continue
                    sil = fill_silhouette(mask)
                    if (sil > 0).sum() < 40:
                        continue
                    bg_lum = V._lum(bg)
                    # светлая/тёмная клетка по координате
                    if (row + col) % 2 == 0:
                        light_lum, dark_lum = 0, 1  # placeholder
                    for name, fn in METHODS.items():
                        v = fn(patch, sil, bg_lum, 0, 0)
                        all_vals[name][0 if true_color == WHITE else 1].append(v)
                        guess = WHITE if v > 105 else BLACK
                        rows[name].append(guess == true_color)
                        if name == "median_core2":
                            img_vals.append((v, true_color))
                            if (v > 75) != (true_color == WHITE):
                                err_details.append((si,
                                                    fen[:20], ch_, round(v, 1),
                                                    (row, col)))
            # адаптивный порог: Оцу по значениям ядра
            if len(img_vals) >= 2:
                vals = np.array([v for v, _ in img_vals], np.float32)
                if vals.max() - vals.min() >= 40:
                    hist, _ = np.histogram(vals, bins=32, range=(0, 256))
                    hist = hist.astype(np.float64)
                    total = hist.sum()
                    sum_all = np.dot(hist, np.arange(32))
                    sumB = 0.0
                    wB = 0.0
                    best_t, best_var = 0, -1.0
                    for i in range(32):
                        wB += hist[i]
                        if wB == 0:
                            continue
                        wF = total - wB
                        if wF == 0:
                            break
                        sumB += i * hist[i]
                        mB = sumB / wB
                        mF = (sum_all - sumB) / wF
                        var = wB * wF * (mB - mF) ** 2
                        if var > best_var:
                            best_var, best_t = var, i
                    thr = (best_t + 0.5) * 256 / 32
                else:
                    thr = 105.0
                for v, tc in img_vals:
                    adapt_rows.append((WHITE if v > thr else BLACK) == tc)

    print(f"{'method':22} {'acc':>6} {'errors':>7}")
    for name, res in rows.items():
        acc = sum(res) / len(res) if res else 0
        print(f"{name:22} {acc:6.3f} {len(res) - sum(res):7d} / {len(res)}")

    # распределения + лучший порог
    print("\nthreshold sweep:")
    for name in METHODS:
        wv, bv = all_vals[name]
        if not wv:
            continue
        best_t, best = 0, 0
        for t in range(0, 256, 1):
            ok = sum(1 for v in wv if v > t) + sum(1 for v in bv if v <= t)
            if ok > best:
                best, best_t = ok, t
        wv.sort()
        bv.sort()
        print(f"{name:22} best_thr={best_t:3d} acc={best / (len(wv) + len(bv)):.3f} "
              f"white[p10,p50,p90]={np.percentile(wv, 10):.0f},"
              f"{np.percentile(wv, 50):.0f},{np.percentile(wv, 90):.0f} "
              f"black={np.percentile(bv, 10):.0f},{np.percentile(bv, 50):.0f},"
              f"{np.percentile(bv, 90):.0f}")


    if adapt_rows:
        print(f"\ncore2 + adaptive thr: acc={sum(adapt_rows) / len(adapt_rows):.4f} "
              f"errors={len(adapt_rows) - sum(adapt_rows)}/{len(adapt_rows)}")

    # ошибки фиксированного порога 75
    print("\nerrors of core2@75:")
    for d in err_details:
        print("   ", d)


if __name__ == "__main__":
    run()
