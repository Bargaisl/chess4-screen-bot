"""Шаблоны фигур: системные шрифты Windows + встроенный набор эталонных силуэтов
для точного распознавания на шахматных платформах (Lichess, Chess.com).
"""
from __future__ import annotations

import os
import glob

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .board import (BISHOP, BLACK, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE)

CANVAS = 48
MARGIN = 4

GLYPHS = {
    (WHITE, PAWN): "\u2659",
    (WHITE, KNIGHT): "\u2658",
    (WHITE, BISHOP): "\u2657",
    (WHITE, ROOK): "\u2656",
    (WHITE, QUEEN): "\u2655",
    (WHITE, KING): "\u2654",
    (BLACK, PAWN): "\u265F",
    (BLACK, KNIGHT): "\u265E",
    (BLACK, BISHOP): "\u265D",
    (BLACK, ROOK): "\u265C",
    (BLACK, QUEEN): "\u265B",
    (BLACK, KING): "\u265A",
}

PIECE_NAME = {PAWN: "пешка", KNIGHT: "конь", BISHOP: "слон",
              ROOK: "ладья", QUEEN: "ферзь", KING: "король"}


def normalize_mask(mask: np.ndarray, canvas: int = CANVAS,
                   margin: int = MARGIN) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((canvas, canvas), np.uint8)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = mask[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    inner = canvas - 2 * margin
    scale = min(inner / max(w, 1), inner / max(h, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((canvas, canvas), np.uint8)
    oy, ox = (canvas - nh) // 2, (canvas - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = (resized > 127).astype(np.uint8) * 255
    return out


def fill_silhouette(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8) * 255
    if m.max() == 0:
        return m
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return m
    out = np.zeros_like(m)
    areas = sorted((cv2.contourArea(c) for c in contours), reverse=True)
    limit = areas[0] * 0.15 if len(areas) > 1 else 0
    for c in contours:
        if cv2.contourArea(c) >= max(limit, 8):
            cv2.drawContours(out, [c], -1, 255, -1)
    return out


def font_has_glyph(font_path: str, ch: str) -> bool:
    font = _load_font(font_path, 64)
    if font is None:
        return False
    probe = "\uE123"
    try:
        a = np.array(font.getmask(ch))
        b = np.array(font.getmask(probe))
    except Exception:
        return False
    if a.size == 0:
        return False
    if b.size and a.shape == b.shape and np.array_equal(a, b):
        return False
    return int((a > 32).sum()) > 24


_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont | None] = {}


def _load_font(path: str, px: int):
    key = (path, px)
    if key not in _FONT_CACHE:
        try:
            _FONT_CACHE[key] = ImageFont.truetype(path, px)
        except Exception:
            _FONT_CACHE[key] = None
    return _FONT_CACHE[key]


_GLYPH_CACHE: dict[tuple[str, str], bool] = {}


def _has_glyph(path: str, ch: str) -> bool:
    key = (path, ch)
    if key not in _GLYPH_CACHE:
        _GLYPH_CACHE[key] = font_has_glyph(path, ch)
    return _GLYPH_CACHE[key]


def render_glyph(ch: str, font_path: str, px: int = 96) -> np.ndarray | None:
    font = _load_font(font_path, px)
    if font is None:
        return None
    canvas_px = px * 3
    img = Image.new("L", (canvas_px, canvas_px), 0)
    draw = ImageDraw.Draw(img)
    draw.text((canvas_px // 3, canvas_px // 3), ch, font=font, fill=255, anchor="mm")
    arr = np.array(img)
    mask = (arr > 96).astype(np.uint8) * 255
    if mask.max() == 0:
        return None
    return mask


def _candidate_fonts(limit: int = 14) -> list[str]:
    dirs = []
    windir = os.environ.get("WINDIR", r"C:\Windows")
    dirs.append(os.path.join(windir, "Fonts"))
    dirs.append(os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft",
                             "Windows", "Fonts"))
    priority = ["seguisym.ttf", "ARIALUNI.ttf", "segoeuib.ttf", "arialbd.ttf",
                "segoeui.ttf", "arial.ttf", "timesbd.ttf", "times.ttf"]
    files = []
    for d in dirs:
        if d and os.path.isdir(d):
            for name in priority:
                p = os.path.join(d, name)
                if os.path.isfile(p):
                    files.append(p)
            for p in sorted(glob.glob(os.path.join(d, "*.ttf"))):
                if p not in files:
                    files.append(p)
    seen, out = set(), []
    for p in files:
        key = os.path.basename(p).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
        if len(out) >= 30:
            break
    return out


class TemplateBank:
    """Силуэты фигур из системных шрифтов + встроенных шаблонов платформ."""

    def __init__(self, max_fonts: int = 8):
        self.templates: dict[tuple[int, int], list[np.ndarray]] = {}
        self.fonts_used: list[str] = []
        self._build_from_stock()
        self._build_from_fonts(max_fonts)
        self._bin: dict[tuple[int, int], np.ndarray] = {}
        self._ink: dict[tuple[int, int], np.ndarray] = {}
        for key, variants in self.templates.items():
            stack = np.stack([(v > 0) for v in variants])
            self._bin[key] = stack
            self._ink[key] = stack.sum(axis=(1, 2))

    def _build_from_stock(self):
        """Загрузка силуэтов стандартных шахматных платформ из каталога templates_stock."""
        base = os.path.dirname(os.path.abspath(__file__))
        stock_dir = os.path.join(base, "templates_stock")
        if not os.path.isdir(stock_dir):
            return
        t_map = {'p': PAWN, 'n': KNIGHT, 'b': BISHOP, 'r': ROOK, 'q': QUEEN, 'k': KING}
        for name in os.listdir(stock_dir):
            if not name.lower().endswith('.png'):
                continue
            base_code = os.path.splitext(name)[0].lower()
            code = base_code[-2:]
            if code[0] not in 'wb' or code[1] not in t_map:
                continue
            color = WHITE if code[0] == 'w' else BLACK
            t = t_map[code[1]]
            p = os.path.join(stock_dir, name)
            try:
                data = np.fromfile(p, dtype=np.uint8)
                tpl_img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            except Exception:
                continue
            if tpl_img is None or tpl_img.size == 0:
                continue
            k = max(2, int(min(tpl_img.shape[:2]) * 0.12))
            corners = np.concatenate([tpl_img[:k, :k].ravel(), tpl_img[:k, -k:].ravel(),
                                      tpl_img[-k:, :k].ravel(), tpl_img[-k:, -k:].ravel()])
            bg = float(np.median(corners))
            diff = cv2.absdiff(tpl_img, np.full_like(tpl_img, int(bg)))
            mask = (diff > 25).astype(np.uint8) * 255
            sil = fill_silhouette(mask)
            norm = normalize_mask(sil)
            if (norm > 0).sum() >= 40:
                self.templates.setdefault((color, t), []).append(norm)

    def _build_from_fonts(self, max_fonts: int):
        fonts = _candidate_fonts()
        used = 0
        for path in fonts:
            if used >= max_fonts:
                break
            glyphs_here = {k: v for k, v in GLYPHS.items() if _has_glyph(path, v)}
            if len(glyphs_here) < 6:
                continue
            used += 1
            self.fonts_used.append(os.path.basename(path))
            for key, ch in glyphs_here.items():
                mask = render_glyph(ch, path)
                if mask is None:
                    continue
                sil = fill_silhouette(mask)
                norm = normalize_mask(sil)
                if (norm > 0).sum() < 40:
                    continue
                self.templates.setdefault(key, []).append(norm)

    def classes(self):
        return self.templates.keys()

    def match_silhouette(self, sil: np.ndarray, color: int | None = None,
                         rel_h: float | None = None) -> tuple[int | None, float]:
        norm = normalize_mask(sil)
        a = norm > 0
        ink_a = int(a.sum())
        if ink_a < 20:
            return None, 0.0

        # Геометрический анализ пропорций силуэта:
        # У пешки круглая голова (верх узкий), у ладьи — широкие зубцы крепости
        ys, xs = np.nonzero(norm)
        h = ys.max() - ys.min() + 1
        y_top = ys.min() + max(1, int(0.20 * h))
        y_base = ys.max() - max(1, int(0.15 * h))
        w_top = float((norm[y_top, :] > 0).sum())
        w_base = float(max(1, (norm[y_base, :] > 0).sum()))
        ratio_top = w_top / w_base

        best_type, best_score = None, 0.0

        for t in (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING):
            keys = ([(color, t)] if color is not None
                    else [(WHITE, t), (BLACK, t)])
            raw_score = 0.0
            for key in keys:
                vb = self._bin.get(key)
                if vb is None:
                    continue
                inter = np.logical_and(vb, a).sum(axis=(1, 2))
                union = ink_a + self._ink[key] - inter
                sc = float(np.max(inter / union))
                if sc > raw_score:
                    raw_score = sc

            score = raw_score

            # Геометрическая защита от путаницы фигур по форме и высоте:
            if score > 0.20:
                # 1. Защита по относительной высоте фигуры (rel_h = h_sil / cell_h):
                if rel_h is not None:
                    if t == PAWN and rel_h > 0.82:
                        score -= 0.35    # пешка не может быть чрезмерно высокой (> 82% клетки)
                    elif t in (BISHOP, QUEEN, KING) and rel_h < 0.58:
                        score -= 0.35    # слон, ферзь и король не бывают карликовыми (< 58% клетки)

                # 2. Защита пешка <-> ладья по ширине макушки (зубцы крепости vs круглая голова):
                if t == ROOK and ratio_top < 0.45:
                    score -= 0.35    # у ладьи не может быть узкой круглой головы (зубцы широкие)
                elif t == PAWN and ratio_top > 0.68:
                    score -= 0.35    # у пешки не может быть широких крепостных зубцов

                # 3. Защита ферзь <-> король:
                if t == KING and ratio_top > 0.68:
                    score -= 0.35    # у короля крестик на макушке, верх узкий
                elif t == QUEEN and ratio_top < 0.50:
                    score -= 0.35    # у ферзя широкая многозубчатая корона


            if score > best_score:
                best_type, best_score = t, score

        return best_type, best_score


_DEFAULT_BANK: TemplateBank | None = None


def default_bank() -> TemplateBank:
    global _DEFAULT_BANK
    if _DEFAULT_BANK is None:
        _DEFAULT_BANK = TemplateBank()
    return _DEFAULT_BANK


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a > 0
    bb = b > 0
    inter = int(np.logical_and(aa, bb).sum())
    union = int(np.logical_or(aa, bb).sum())
    return inter / union if union else 0.0
