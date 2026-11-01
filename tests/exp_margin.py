"""Насколько уверенно сопоставление отличает типы фигур друг от друга.

Считает для каждой занятой клетки скор каждого класса (по всем вариантам
шаблонов) и смотрит на ошибки и «зазор» между правильным и лучшим чужим
классом. Сравнивает метрики: IoU (текущая), корреляция, IoU по контуру.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2

from exp_color2 import STYLES, FENS, expand
from chessvision import vision as V
from chessvision.templates import (default_bank, fill_silhouette,
                                   normalize_mask)
from chessvision.board import (BISHOP, KING, KNIGHT, PAWN, QUEEN, ROOK,
                               PIECE_CHARS)

TYPES = (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)
NAMES = {PAWN: "P", KNIGHT: "N", BISHOP: "B", ROOK: "R", QUEEN: "Q", KING: "K"}


def iou(a, b):
    aa, bb = a > 0, b > 0
    u = np.logical_or(aa, bb).sum()
    return float(np.logical_and(aa, bb).sum()) / u if u else 0.0


def corr(a, b):
    x = (a > 0).astype(np.float32).ravel()
    y = (b > 0).astype(np.float32).ravel()
    x = x - x.mean()
    y = y - y.mean()
    d = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(x @ y) / d if d else 0.0


def outline(mask):
    m = (mask > 0).astype(np.uint8) * 255
    er = cv2.erode(m, np.ones((3, 3), np.uint8))
    return ((m > 0) & ~(er > 0)).astype(np.uint8) * 255


def fill_holes(mask):
    """Сплошная заливка внешнего контура (как у силуэта с экрана)."""
    m = (mask > 0).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    if cnts:
        cv2.drawContours(out, cnts, -1, 255, -1)
    return out


def make_metrics(bank):
    """Заранее готовит варианты каждого класса под каждую метрику."""
    data = {"iou": {}, "corr": {}, "edge": {}, "ifill": {}}
    for key, variants in bank.templates.items():
        t = key[1]
        data["iou"].setdefault(t, []).extend(variants)
        data["corr"].setdefault(t, []).extend(variants)
        data["edge"].setdefault(t, []).extend(outline(v) for v in variants)
        data["ifill"].setdefault(t, []).extend(fill_holes(v)
                                               for v in variants)
    return data


def score_all(norm, prepared, metric):
    out = {}
    for t, variants in prepared[metric].items():
        fn = corr if metric == "corr" else iou
        out[t] = max(fn(norm, v) for v in variants)
    return out


def collect(bank, prepared):
    """-> список (метка, словарь класс->скор, rect, style, fen, cell)."""
    samples = []
    for style in STYLES:
        for fen in FENS:
            want = fen.split()[0]
            from test_vision import render_board
            img = render_board(want, **style)
            rect = V.detect_board(img)
            if rect is None:
                continue
            x, y, w, h = rect
            cw, ch = w / 8.0, h / 8.0
            exp = expand(want)
            for row in range(8):
                for col in range(8):
                    wc = exp[row][col]
                    if wc == "1":
                        continue
                    x0 = int(round(col * cw)) + V.CELL_INSET
                    y0 = int(round(row * ch)) + V.CELL_INSET
                    x1 = int(round((col + 1) * cw)) - V.CELL_INSET
                    y1 = int(round((row + 1) * ch)) - V.CELL_INSET
                    patch = img[y + y0:y + y1, x + x0:x + x1]
                    if patch.size == 0 or min(patch.shape[:2]) < 6:
                        continue
                    mask = V._cell_mask(patch, V._median_bg(patch))
                    if float((mask > 0).mean()) > 0.72:
                        continue
                    sil = fill_silhouette(mask)
                    if (sil > 0).sum() < 40:
                        continue
                    norm = normalize_mask(sil)
                    entry = {"true": CHAR_T[wc.upper()], "rect": rect,
                             "cell": (row, col), "style": style["cell_px"]}
                    for m in ("iou", "corr", "edge", "ifill"):
                        entry[m] = score_all(norm, prepared, m)
                    samples.append(entry)
    return samples


CHAR_T = {"P": PAWN, "N": KNIGHT, "B": BISHOP, "R": ROOK, "Q": QUEEN,
          "K": KING}


def report(samples):
    print(f"cells={len(samples)}")
    for m in ("iou", "ifill", "corr", "edge"):
        wrong, margins, pair = 0, [], []
        for s in samples:
            scores = s[m]
            true_t = s["true"]
            best_t = max(scores, key=lambda k: scores[k])
            if best_t != true_t:
                wrong += 1
                pair.append((NAMES[true_t], NAMES[best_t],
                             round(scores[true_t], 3),
                             round(scores[best_t], 3), s["cell"]))
            others = [v for k, v in scores.items() if k != true_t]
            margins.append(scores[true_t] - max(others))
        margins = sorted(margins)
        print(f"  {m:5} errors={wrong:3d}  margin p5..p50=" 
              f"{[round(float(np.percentile(margins, p)), 3) for p in (5, 25, 50, 75)]}")
        for p in pair[:15]:
            print("      conf", p)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    bank = default_bank()
    prepared = make_metrics(bank)
    report(collect(bank, prepared))
