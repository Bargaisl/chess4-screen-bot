"""Съёмка экрана, поиск шахматной доски и надёжное распознавание позиции.

Улучшения версии chess4:
1. Защита от сдвига сетки: строгое соблюдение цветового инварианта углов
   (клетка a1 всегда тёмная, a8/h1 всегда светлые). Никаких смещений на 1 клетку!
2. Физическая высота фигур: классификатор учитывает отношение высоты фигуры
   к высоте клетки (пешки не путаются со слонами и ферзями).
3. Поддержка выбора монитора (мультимониторность).
"""
from __future__ import annotations

import bisect
import threading

import numpy as np
import cv2
import mss

from .board import BLACK, WHITE, PIECE_CHARS
from .templates import TemplateBank, default_bank, fill_silhouette

MASK_T = 42
MIN_SCORE = 0.30
GOOD_MATCH = 0.88
RING = 0.16
CELL_INSET = 2

_LOCAL = threading.local()


def _sct():
    sct = getattr(_LOCAL, "sct", None)
    if sct is None:
        sct = mss.mss()
        _LOCAL.sct = sct
    return sct


def capture(region: dict | None = None, monitor_idx: int = 1) -> np.ndarray:
    """Снимок экрана с выбранного монитора -> BGR-массив."""
    sct = _sct()
    if region is not None:
        mon = region
    else:
        # 0 = все мониторы, 1 = первый, 2 = второй...
        idx = max(0, min(monitor_idx, len(sct.monitors) - 1))
        mon = sct.monitors[idx]
    shot = sct.grab(mon)
    img = np.array(shot)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return np.ascontiguousarray(img)


# ------------------------------------------------------------- поиск доски
def _cluster_lines(items, tol: float):
    if not items:
        return []
    items = sorted(items)
    groups, cur = [], [items[0]]
    for pos, w in items[1:]:
        if pos - cur[-1][0] <= tol:
            cur.append((pos, w))
        else:
            groups.append(cur)
            cur = [(pos, w)]
    groups.append(cur)
    out = []
    for g in groups:
        tw = sum(w for _, w in g) or 1.0
        out.append((sum(p * w for p, w in g) / tw, tw))
    return out


def _axis_candidates(pos_w, n: int, min_step: float, max_step: float,
                     keep: int = 6):
    out = []
    m = len(pos_w)
    if m < 3:
        return out
    xs = [p for p, _ in pos_w]
    best = {}
    for i in range(m):
        pi, wi = pos_w[i]
        for j in range(i + 1, m):
            pj, wj = pos_w[j]
            d = pj - pi
            for k in range(1, n + 1):
                s = d / k
                if s < min_step or s > max_step:
                    continue
                tol = max(2.0, s * 0.13)
                for t0 in (0, 1, 2):
                    a = pi - t0 * s
                    if a < 0:
                        continue
                    matched, weight = 0, 0.0
                    for t in range(n + 1):
                        q = a + t * s
                        idx = bisect.bisect_left(xs, q - tol)
                        if idx < m and xs[idx] <= q + tol:
                            matched += 1
                            weight += pos_w[idx][1]
                    if matched < n - 1:
                        continue
                    score = matched * 1e6 + weight
                    key = (round(a / 3), round(s * 8))
                    if key not in best or score > best[key][0]:
                        best[key] = (score, a, a + n * s, s, matched)
    out = sorted(best.values(), key=lambda x: -x[0])
    diverse = []
    for c in out:
        _, a, _b, s, _m = c
        if any(abs(a - a2) <= max(4.0, 0.3 * s) and
               abs(s - s2) <= max(2.0, 0.05 * s) for _, a2, _b2, s2, _m2
               in diverse):
            continue
        diverse.append(c)
        if len(diverse) >= keep:
            break
    return diverse


FLAT_THR = 28.0
GAP_MIN = 15.0
PAR_MIN = 0.80
BG_PAR_MIN = 0.85
FINE_POOL = 12


def _cell_bg(img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    k = max(2, int(min(y1 - y0, x1 - x0) * 0.28))
    parts = [img[y0:y0 + k, x0:x0 + k], img[y0:y0 + k, x1 - k:x1],
             img[y1 - k:y1, x0:x0 + k], img[y1 - k:y1, x1 - k:x1]]
    pix = np.concatenate([p[::2, ::2].reshape(-1, 3) for p in parts], axis=0)
    return np.median(pix, axis=0).astype(np.float32)


def _bg_parity_score(img: np.ndarray, x: float, y: float,
                     sx: float, sy: float) -> float:
    """Проверка фона всех 64 клеток + защита от сдвига на 1 клетку."""
    bg = np.zeros((8, 8, 3), np.float32)
    for row in range(8):
        for col in range(8):
            x0 = int(round(x + col * sx)) + 1
            y0 = int(round(y + row * sy)) + 1
            x1 = int(round(x + (col + 1) * sx)) - 1
            y1 = int(round(y + (row + 1) * sy)) - 1
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
            if x1 - x0 < 6 or y1 - y0 < 6:
                return -1e9
            bg[row, col] = _cell_bg(img, x0, y0, x1, y1)

    par = np.zeros((8, 8), bool)
    for row in range(8):
        for col in range(8):
            par[row, col] = ((row + col) % 2) == 0

    m0 = np.median(bg[par], axis=0)
    m1 = np.median(bg[~par], axis=0)
    gap = float(np.linalg.norm(m0 - m1))
    if gap < GAP_MIN:
        return -1e9

    # ИНВАРИАНТ ШАХМАТНОЙ ДОСКИ:
    # Клетки (row + col) % 2 == 0 ОБЯЗАНЫ быть СВЕТЛЫМИ!
    # Если m0 темнее m1 — сетка смещена на 1 клетку в сторону, немедленный отказ!
    lum0 = _lum(m0)
    lum1 = _lum(m1)
    if lum0 <= lum1 + 8.0:
        return -1e9

    d0 = np.linalg.norm(bg - m0, axis=2)
    d1 = np.linalg.norm(bg - m1, axis=2)
    agree = float(((d0 < d1) == par).mean())
    if agree < BG_PAR_MIN:
        return -1e9
    resid = float(np.minimum(d0, d1).mean())
    return agree * 100.0 - resid * 0.3


def _line_contrast(img: np.ndarray, x: float, y: float,
                   sx: float, sy: float) -> float:
    h, w = img.shape[:2]
    total, n = 0.0, 0
    xa, xb = int(round(x)), int(round(x + 8 * sx))
    ya, yb = int(round(y)), int(round(y + 8 * sy))
    ya, yb = max(0, ya), min(h, yb)
    xa, xa2 = max(0, xa), min(w, xb)
    for k in range(9):
        xc = int(round(x + k * sx))
        if xc - 3 < 0 or xc + 3 > w:
            continue
        left = img[ya:yb, xc - 3:xc - 1].astype(np.float32)
        right = img[ya:yb, xc + 1:xc + 3].astype(np.float32)
        if left.size and right.size == left.size:
            total += float(np.abs(left - right).mean())
            n += 1
    for k in range(9):
        yc = int(round(y + k * sy))
        if yc - 3 < 0 or yc + 3 > h:
            continue
        up = img[yc - 3:yc - 1, xa:xa2].astype(np.float32)
        down = img[yc + 1:yc + 3, xa:xa2].astype(np.float32)
        if up.size and down.size == up.size:
            total += float(np.abs(up - down).mean())
            n += 1
    return total / n if n else 0.0


def _alternation_score(img: np.ndarray, x: float, y: float,
                       sx: float, sy: float) -> float:
    means = np.zeros((8, 8, 3), np.float32)
    spread = np.zeros((8, 8), np.float32)
    for row in range(8):
        for col in range(8):
            x0 = int(round(x + col * sx)) + 1
            y0 = int(round(y + row * sy)) + 1
            x1 = int(round(x + (col + 1) * sx)) - 1
            y1 = int(round(y + (row + 1) * sy)) - 1
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
            if x1 - x0 < 3 or y1 - y0 < 3:
                return -1e9
            patch = img[y0:y1:2, x0:x1:2].reshape(-1, 3)
            means[row, col] = patch.mean(axis=0)
            spread[row, col] = float(patch.std(axis=0).mean())
    diffs = []
    for row in range(8):
        for col in range(8):
            if col < 7:
                diffs.append(float(np.linalg.norm(means[row, col] - means[row, col + 1])))
            if row < 7:
                diffs.append(float(np.linalg.norm(means[row, col] - means[row + 1, col])))
    diffs = np.array(diffs)
    good = float((diffs > 18).mean())
    if good < 0.85:
        return -1e9

    par = np.zeros((8, 8), bool)
    for row in range(8):
        for col in range(8):
            par[row, col] = ((row + col) % 2) == 0
    empty = spread < FLAT_THR
    n0 = int((empty & par).sum())
    n1 = int((empty & ~par).sum())
    parity_bonus = 0.0
    if n0 >= 8 and n1 >= 8:
        m0 = np.median(means[empty & par], axis=0)
        m1 = np.median(means[empty & ~par], axis=0)
        gap = float(np.linalg.norm(m0 - m1))
        if gap < GAP_MIN:
            return -1e9

        # Инвариант чётности: чётные клетки должны быть светлее нечётных
        if _lum(m0) <= _lum(m1) + 6.0:
            return -1e9

        d0 = np.linalg.norm(means - m0, axis=2)
        d1 = np.linalg.norm(means - m1, axis=2)
        agree = float((((d0 < d1) == par) & empty).sum()) / (n0 + n1)
        if agree < PAR_MIN:
            return -1e9
        parity_bonus = 60.0

    flat = means.reshape(-1, 3)
    order = np.argsort(flat @ np.array([0.299, 0.587, 0.114], np.float32))
    lo = flat[order[:32]].mean(axis=0)
    hi = flat[order[32:]].mean(axis=0)
    split = float(np.linalg.norm(lo - hi))
    contrast = _line_contrast(img, x, y, sx, sy)
    return (good * 100.0 + split - float(np.median(spread)) * 0.5
            + parity_bonus + 3.0 * contrast)


def detect_board(img: np.ndarray, min_cell: int = 28,
                 max_cell: int = 400) -> tuple[int, int, int, int] | None:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 130)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=55,
                            minLineLength=max(36, int(0.18 * min(h, w))),
                            maxLineGap=10)
    if lines is None:
        return None
    hor, ver = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = abs(int(x2) - int(x1)), abs(int(y2) - int(y1))
        if dx <= 3 and dy <= 3:
            continue
        if dy <= 0.3 * dx:
            hor.append(((y1 + y2) / 2.0, dx))
        elif dx <= 0.3 * dy:
            ver.append(((x1 + x2) / 2.0, dy))
    cand_y = _axis_candidates(_cluster_lines(hor, 4), 8, min_cell, max_cell, keep=16)
    cand_x = _axis_candidates(_cluster_lines(ver, 4), 8, min_cell, max_cell, keep=16)
    if not cand_y or not cand_x:
        return None

    pool = []
    for _, y0, y1, sy, my in cand_y:
        for _, x0, x1, sx, mx in cand_x:
            ratio = sx / sy if sy else 99
            if not (0.88 <= ratio <= 1.14):
                continue
            if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
                continue
            score = _alternation_score(img, x0, y0, sx, sy)
            if score <= -1e8:
                continue
            score += 10.0 * (my + mx)
            pool.append((score, x0, y0, sx, sy, x1 - x0, y1 - y0))
    if not pool:
        return None

    pool.sort(key=lambda r: -r[0])
    best, best_score = None, -1e18
    for score, x0, y0, sx, sy, bw, bh in pool[:FINE_POOL]:
        fine = _bg_parity_score(img, x0, y0, sx, sy)
        if fine <= -1e8:
            continue
        total = fine + score
        if total > best_score:
            best_score = total
            best = (x0, y0, bw, bh)
    if best is None or best_score <= 0:
        return None
    return (int(round(best[0])), int(round(best[1])),
            int(round(best[2])), int(round(best[3])))


def snap_selection(img: np.ndarray, rect: tuple[int, int, int, int] | list[int],
                   margin: int = 36) -> tuple[int, int, int, int]:
    """Автоматическая привязка пользовательского выделения к истинной шахматной сетке 8x8."""
    x, y, w, h = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
    H, W = img.shape[:2]
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(W, x + w + margin)
    y1 = min(H, y + h + margin)
    if (x1 - x0) < 64 or (y1 - y0) < 64:
        return (x, y, w, h)

    crop = img[y0:y1, x0:x1]
    det = detect_board(crop, min_cell=max(16, int(w / 12)), max_cell=max(36, int(w / 6)))
    if det is not None:
        dx, dy, dw, dh = det
        cand_x = x0 + dx
        cand_y = y0 + dy
        if (abs(cand_x - x) <= margin and abs(cand_y - y) <= margin
                and abs(dw - w) <= margin * 2 and abs(dh - h) <= margin * 2):
            return (cand_x, cand_y, dw, dh)
    return (x, y, w, h)


# --------------------------------------------------------- распознавание
def _board_colors(cells_mean: np.ndarray):
    lum = cells_mean @ np.array([0.299, 0.587, 0.114], np.float32)
    c_hi = cells_mean[int(np.argmax(lum))].astype(np.float32)
    c_lo = cells_mean[int(np.argmin(lum))].astype(np.float32)
    data = cells_mean.astype(np.float32)
    for _ in range(6):
        d_hi = np.linalg.norm(data - c_hi, axis=1)
        d_lo = np.linalg.norm(data - c_lo, axis=1)
        m_hi = d_hi <= d_lo
        if m_hi.any():
            c_hi = data[m_hi].mean(axis=0)
        if (~m_hi).any():
            c_lo = data[~m_hi].mean(axis=0)
    return c_hi, c_lo


def _corner_bg(patch: np.ndarray) -> np.ndarray:
    ph, pw = patch.shape[:2]
    k = max(2, int(round(min(ph, pw) * RING)))
    corners = [patch[:k, :k], patch[:k, -k:], patch[-k:, :k], patch[-k:, -k:]]
    vals = np.stack([c.reshape(-1, 3).mean(axis=0) for c in corners])
    return np.median(vals, axis=0)


def _median_bg(patch: np.ndarray) -> np.ndarray:
    return np.median(patch.reshape(-1, 3), axis=0)


def _cell_mask(patch: np.ndarray, bg: np.ndarray) -> np.ndarray:
    p = patch if patch.dtype == np.float32 else patch.astype(np.float32)
    d = p - bg.reshape(1, 1, 3)
    m = (((d * d).sum(axis=2)) > MASK_T * MASK_T).astype(np.uint8) * 255
    h, w = m.shape[:2]
    if (m[0, :] > 0).mean() > 0.8: m[0, :] = 0
    if (m[h - 1, :] > 0).mean() > 0.8: m[h - 1, :] = 0
    if (m[:, 0] > 0).mean() > 0.8: m[:, 0] = 0
    if (m[:, w - 1] > 0).mean() > 0.8: m[:, w - 1] = 0
    return m


def _lum(c) -> float:
    return float(0.299 * c[2] + 0.587 * c[1] + 0.114 * c[0])


DEFAULT_LUM_THR = 95.0
SQ_COEFF = 0.7
SQ_BIAS = 15.0
BD_THR = -0.1


def _is_white(med: float, sq_lum: float, bd: float) -> bool:
    if med > DEFAULT_LUM_THR:
        return True
    return med > SQ_COEFF * sq_lum + SQ_BIAS and bd > BD_THR


def _classify(patch: np.ndarray, bank: TemplateBank, global_colors,
              cell_h: float | None = None):
    if patch.size == 0 or min(patch.shape[:2]) < 6:
        return None, 0.0, 0.0, 0.0
    pf = patch if patch.dtype == np.float32 else patch.astype(np.float32)
    bgs = [_median_bg(pf), _corner_bg(pf)]
    if global_colors is not None:
        bgs.extend([global_colors[0], global_colors[1]])
    best = (None, 0.0, 0.0, 0.0)
    for bg in bgs:
        mask = _cell_mask(pf, bg)
        ink = float((mask > 0).mean())
        if ink > 0.72 or ink < 0.08:
            continue
        sil = fill_silhouette(mask)
        if (sil > 0).sum() < 40:
            continue

        ys, xs = np.nonzero(sil)
        h_sil = float(ys.max() - ys.min() + 1) if len(ys) else 0.0
        rel_h = (h_sil / cell_h) if (cell_h and cell_h > 0) else None

        ptype, score = bank.match_silhouette(sil, rel_h=rel_h)
        if ptype is not None and score > best[1]:
            med, bd = _core_stats(patch, sil)
            best = (ptype, score, med, bd)
            if score >= GOOD_MATCH:
                break
    if best[0] is None or best[1] < MIN_SCORE:
        return None, best[1], best[2], best[3]
    return best


def _core(silhouette: np.ndarray) -> np.ndarray:
    core = cv2.erode((silhouette > 0).astype(np.uint8) * 255,
                     np.ones((5, 5), np.uint8))
    if (core > 0).sum() < 20:
        core = silhouette
    return core


def _core_stats(patch: np.ndarray, silhouette: np.ndarray):
    core = _core(silhouette)
    ys, xs = np.nonzero(core)
    if len(xs) == 0:
        return 0.0, 0.0
    region = patch[ys, xs].astype(np.float32).reshape(-1, 3)
    lum = region @ np.array([0.114, 0.587, 0.299], np.float32)
    bd = float((lum > 150).mean()) - float((lum < 40).mean())
    return float(np.median(lum)), bd


def read_board(img: np.ndarray, rect: tuple[int, int, int, int],
               orientation: str = "auto",
               bank: TemplateBank | None = None,
               my_color_hint: str = "white") -> dict:
    if bank is None:
        bank = default_bank()
    x, y, w, h = rect
    H, W = img.shape[:2]
    x, y = max(0, int(x)), max(0, int(y))
    w, h = int(w), int(h)
    w, h = min(w, W - x), min(h, H - y)
    if w < 8 or h < 8:
        return {"placement": None, "confidence": 0.0, "error": "пустая область"}

    board = img[y:y + h, x:x + w]
    cw, ch = w / 8.0, h / 8.0
    patches = {}
    means = np.zeros((8, 8, 3), np.float32)
    for row in range(8):
        for col in range(8):
            x0 = int(round(col * cw)) + CELL_INSET
            y0 = int(round(row * ch)) + CELL_INSET
            x1 = int(round((col + 1) * cw)) - CELL_INSET
            y1 = int(round((row + 1) * ch)) - CELL_INSET
            patch = board[y0:y1, x0:x1]
            if patch.size == 0 or min(patch.shape[:2]) < 6:
                continue
            patches[(col, row)] = patch
            means[row, col] = patch.reshape(-1, 3).mean(axis=0)

    c_hi, c_lo = _board_colors(means.reshape(-1, 3))
    glob = (c_hi, c_lo)

    cells_info = {}
    for key, patch in patches.items():
        ptype, score, med, bd = _classify(patch, bank, glob, cell_h=ch)
        sq_lum = _lum(_median_bg(patch))
        cells_info[key] = (ptype, score, med, sq_lum, bd)

    scores = []
    rows_out = []
    for row in range(8):
        chars = []
        for col in range(8):
            info = cells_info.get((col, row))
            if info is None:
                chars.append("1")
                continue
            ptype, score, med, sq_lum, bd = info
            if ptype is None:
                chars.append("1")
                continue
            color = WHITE if _is_white(med, sq_lum, bd) else BLACK
            chars.append(PIECE_CHARS[color | ptype])
            scores.append(float(score))
        out, empty = "", 0
        for ch_sym in chars:
            if ch_sym == "1":
                empty += 1
            else:
                if empty:
                    out += str(empty)
                    empty = 0
                out += ch_sym
        if empty:
            out += str(empty)
        rows_out.append(out)

    placement = "/".join(rows_out)
    active_orientation = orientation
    if orientation == "auto":
        active_orientation = detect_orientation(rows_out, my_color_hint)

    if active_orientation == "black":
        placement = _flip_placement(placement)
    return {
        "placement": placement,
        "confidence": float(np.mean(scores)) if scores else 0.0,
        "orientation": active_orientation,
        "unknown": 0,
    }


def detect_orientation(raw_rows: list[str], my_color_hint: str = "white") -> str:
    """Определяет ориентацию доски по расположению фигур на экране.
    
    Анализирует положение королей и распределение фигур:
    - Белый король ниже чёрного (или в рядах 4-7) -> белые снизу ('white')
    - Чёрный король ниже белого (или в рядах 4-7) -> чёрные снизу ('black')
    """
    grid = []
    for r in raw_rows:
        expanded = []
        for ch in r:
            if ch.isdigit():
                expanded.extend(["."] * int(ch))
            else:
                expanded.append(ch)
        grid.append(expanded)

    # 1. Положение королей — самый фундаментальный инвариант ориентации в любой фазе игры
    white_king_row = None
    black_king_row = None
    for r in range(len(grid)):
        for ch in grid[r]:
            if ch == "K":
                white_king_row = r
            elif ch == "k":
                black_king_row = r

    if white_king_row is not None and black_king_row is not None and white_king_row != black_king_row:
        return "white" if white_king_row > black_king_row else "black"

    # 2. Если одного из королей нет, проверяем положение оставшегося
    if white_king_row is not None:
        return "white" if white_king_row >= 4 else "black"
    if black_king_row is not None:
        return "black" if black_king_row >= 4 else "white"

    # 3. Сравниваем верхнюю половину (ряды 0-3) и нижнюю (4-7)
    w_top = sum(1 for r in range(4) for ch in grid[r] if ch.isupper())
    b_top = sum(1 for r in range(4) for ch in grid[r] if ch.islower())
    w_bot = sum(1 for r in range(4, 8) for ch in grid[r] if ch.isupper())
    b_bot = sum(1 for r in range(4, 8) for ch in grid[r] if ch.islower())

    score_white_at_bot = b_top + w_bot
    score_black_at_bot = w_top + b_bot

    if score_black_at_bot > score_white_at_bot:
        return "black"
    if score_white_at_bot > score_black_at_bot:
        return "white"

    # 4. Резервный выбор по настройке пользователя
    return "black" if my_color_hint == "black" else "white"


def _flip_placement(placement: str) -> str:
    rows = placement.split("/")
    flipped = []
    for r in reversed(rows):
        expanded = []
        for ch in r:
            if ch.isdigit():
                expanded.extend(["1"] * int(ch))
            else:
                expanded.append(ch)
        expanded.reverse()
        out, empty = "", 0
        for ch in expanded:
            if ch == "1":
                empty += 1
            else:
                if empty:
                    out += str(empty)
                    empty = 0
                out += ch
        if empty:
            out += str(empty)
        flipped.append(out)
    return "/".join(flipped)


def square_to_cell(sq: int, orientation: str = "white") -> tuple[int, int]:
    col = sq & 7
    row = 7 - (sq >> 4)
    if orientation == "black":
        col = 7 - col
        row = 7 - row
    return col, row
