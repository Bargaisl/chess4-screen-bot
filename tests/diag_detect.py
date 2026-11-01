"""Диагностика поиска доски: почему выбран именно этот прямоугольник.

Печатает кластеры линий Хафа, кандидатов по осям и подробности по
топ-гипотезам: счёт чередования, «плоскость» клеток и то, насколько
цвет клетки согласуется с её чётностью (r+c) — настоящая шахматная
доска обязательно чередуется, а тулбар браузера — нет.

python tests/diag_detect.py [rect]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from chessvision import vision as V


def parity_stats(img, x, y, w, h, sx, sy):
    """-> (доля плоских клеток, согласованность чётности среди плоских,
           разрыв яркости двух групп)."""
    means = np.zeros((8, 8, 3), np.float32)
    spread = np.zeros((8, 8), np.float32)
    for r in range(8):
        for c in range(8):
            x0 = int(round(x + c * sx)) + 1
            y0 = int(round(y + r * sy)) + 1
            x1 = int(round(x + (c + 1) * sx)) - 1
            y1 = int(round(y + (r + 1) * sy)) - 1
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
            if x1 - x0 < 3 or y1 - y0 < 3:
                return 0.0, 0.0, 0.0
            p = img[y0:y1, x0:x1].reshape(-1, 3)
            means[r, c] = p.mean(axis=0)
            spread[r, c] = float(p.std(axis=0).mean())
    flat = spread < 28.0
    frac_flat = float(flat.mean())
    par = np.zeros((8, 8), bool)
    for r in range(8):
        for c in range(8):
            par[r, c] = ((r + c) % 2) == 0
    if flat.sum() < 16:
        return frac_flat, 0.0, 0.0
    m0 = np.median(means[flat & par], axis=0)
    m1 = np.median(means[flat & ~par], axis=0)
    gap = float(np.linalg.norm(m0 - m1))
    ok = 0
    n = 0
    for r in range(8):
        for c in range(8):
            if not flat[r, c]:
                continue
            n += 1
            d0 = float(np.linalg.norm(means[r, c] - m0))
            d1 = float(np.linalg.norm(means[r, c] - m1))
            if (d0 < d1) == par[r, c]:
                ok += 1
    return frac_flat, (ok / n if n else 0.0), gap


def main():
    img = V.capture(None)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 130)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=55,
                            minLineLength=max(36, int(0.18 * min(h, w))),
                            maxLineGap=10)
    hor, ver = [], []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            dx, dy = abs(int(x2) - int(x1)), abs(int(y2) - int(y1))
            if dx <= 3 and dy <= 3:
                continue
            if dy <= 0.3 * dx:
                hor.append(((y1 + y2) / 2.0, dx))
            elif dx <= 0.3 * dy:
                ver.append(((x1 + x2) / 2.0, dy))
    hy = V._cluster_lines(hor, 4)
    vx = V._cluster_lines(ver, 4)
    print(f"линий: гориз {len(hor)} -> кластеров {len(hy)}, "
          f"вертик {len(ver)} -> кластеров {len(vx)}")
    print("горизонтальные кластеры (y, вес):",
          [(round(p), round(wt)) for p, wt in hy])
    print("вертикальные  кластеры (x, вес):",
          [(round(p), round(wt)) for p, wt in vx])

    cand_y = V._axis_candidates(hy, 8, 28, 400, keep=16)
    cand_x = V._axis_candidates(vx, 8, 28, 400, keep=16)
    print("\nкандидаты по Y (счёт, начало, конец, шаг, найдено линий):")
    for sc, a, b, s, m in cand_y[:8]:
        print(f"   {round(sc)}  {a:7.1f} {b:7.1f} step={s:6.1f} lines={m}")
    print("кандидаты по X:")
    for sc, a, b, s, m in cand_x[:8]:
        print(f"   {round(sc)}  {a:7.1f} {b:7.1f} step={s:6.1f} lines={m}")

    print("\nгипотезы (топ по итоговому счёту):")
    results = []
    for _, y0, y1, sy, my in cand_y:
        for _, x0, x1, sx, mx in cand_x:
            if not (0.86 <= sx / sy <= 1.16):
                continue
            if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
                continue
            alt = V._alternation_score(img, x0, y0, sx, sy)
            if alt <= -1e8:
                continue
            total = alt + 10.0 * (my + mx)
            fine = V._bg_parity_score(img, x0, y0, sx, sy)
            flat, parity, gap = parity_stats(img, x0, y0, sx * 8, sy * 8,
                                             sx, sy)
            results.append((total + (fine if fine > -1e8 else 0), total, fine,
                            x0, y0, sx * 8, sy * 8, my + mx,
                            flat, parity, gap))
    results.sort(key=lambda r: -r[0])
    for total, cheap, fine, x0, y0, bw, bh, mlines, flat, parity, gap in \
            results[:10]:
        mark = "" if fine > -1e8 else "  ОТКЛЮЧЕНА"
        print(f"  rect=({x0:.0f},{y0:.0f},{bw:.0f},{bh:.0f}) "
              f"итог={total:.1f} дешёв={cheap:.1f} точн={fine:.1f} "
              f"линий={mlines} плоских={flat:.2f} чётность={parity:.2f} "
              f"разрыв={gap:.0f}{mark}")

    got = V.detect_board(img)
    print(f"\nвыбрано: {got}")
    if got:
        x0, y0, bw, bh = got
        flat, parity, gap = parity_stats(img, x0, y0, bw, bh, bw / 8, bh / 8)
        print(f"для выбранного: плоских={flat:.2f} чётность={parity:.2f} "
              f"разрыв={gap:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
