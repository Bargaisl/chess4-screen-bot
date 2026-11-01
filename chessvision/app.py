"""Графический интерфейс и клик-проходимый оверлей поверх экрана (chess4).

Ключевые преимущества версии chess4:
1. Выбор любого монитора (полная поддержка многомониторных систем).
2. Строгая валидация расстановки (защита от диких ходов вроде hxg8=Q).
3. Точное ручное выделение области доски без принудительного ложного авто-сдвига.
4. Кнопки микро-подгонки рамки доски (сдвиг и размер с шагом в 1-2 пикселя).
5. Визуальная проверка сетки 8x8 (кнопка «Сетка» на оверлее).
6. Глобальные горячие клавиши: F2 (Оверлей), F3 (Пересчёт), F4 (Пауза), Esc (Паника).
7. Опциональный человекоподобный автоход (Humanized Click-to-Move).
"""
from __future__ import annotations

import ctypes
import json
import os
import random
import tempfile
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np

from .board import (BLACK, WHITE, Position, board_from_placement, infer_move,
                    placement_issues, side_that_moved, sq_name)
from .engine import Engine
from .vision import capture, detect_board, read_board, snap_selection, square_to_cell

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

DEFAULT_CONFIG = {
    "rect": None,             # [left, top, width, height] или null (автопоиск)
    "monitor": 1,             # индекс монитора (1 = основной, 2 = второй, 0 = все)
    "orientation": "auto",    # auto | white | black
    "my_color": "white",      # white | black | any
    "turn_hint": "white",     # white | black
    "poll_ms": 300,           # период захвата экрана (мс)
    "think_ms": 1200,         # время размышления движка (мс)
    "auto_detect": True,      # автопоиск, если rect не задан
    "enabled": True,          # слежение включено
    "show_grid": False,       # рисовать ли сетку клеток на оверлее
    "automove": False,        # автоход мышью (кликами)
    "multipv": 2,             # число анализируемых линий (1, 2 или 3)
    "elo": 0,                 # 0 = макс (Stockfish), 2500, 2100, 1800, 1500
    "human_time": True,       # динамический человеческий ритм размышления
    "human_mouse": True,      # кривые Безье и задержки мыши (анти-чит)
    "drag_drop": True,        # режим мыши: True = перетаскивание (Drag), False = 2 клика
    "voice": False,           # голосовая озвучка ходов в наушник
}

TRANSPARENT = "#010203"
ARROW = "#00e676"             # Зелёная основная стрелка (#1 лучший ход)
ARROW_SECONDARY = "#ffd600"   # Жёлтая альтернативная стрелка (#2 ход)
ARROW_TERTIARY = "#00b0ff"    # Синяя/бирюзовая стрелка (#3 ход)
HL_FROM = "#ffd400"
HL_TO = "#ff9500"
GRID_COLOR = "#00ff88"
PANEL_BG = "#1c1c1e"

REBUILD_STABILITY = 3
REDETECT_S = 6.0
CONF_MIN = 0.45
CONF_WARN = 0.65


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            saved = json.load(fh)
        if isinstance(saved, dict):
            cfg.update(saved)
    except (OSError, ValueError):
        pass
    if cfg.get("rect") is not None:
        try:
            cfg["rect"] = [int(v) for v in cfg["rect"]]
            if len(cfg["rect"]) != 4:
                cfg["rect"] = None
        except (TypeError, ValueError):
            cfg["rect"] = None
    return cfg


def save_config(cfg: dict) -> None:
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, ensure_ascii=False, indent=2)
    except OSError:
        pass


class Tracker:
    """Следит за позицией на экране с фильтрацией шума."""

    def __init__(self, turn_hint: int = WHITE):
        self.pos = None
        self.turn_hint = turn_hint
        self.pending: tuple[str | None, int] = (None, 0)
        self.seq = 0
        self.last_move = None
        self.confidence = 0.0
        self.manual = False

    def reset(self, placement: str | None = None, turn: int | None = None):
        self.pending = (None, 0)
        self.pos = None
        self.last_move = None
        self.manual = False
        if placement:
            self.update(placement, self.confidence)

    def set_manual(self, pos) -> bool:
        self.pending = (None, 0)
        self.pos = pos
        self.last_move = None
        self.manual = True
        self.seq += 1
        return True

    def set_turn(self, turn: int) -> bool:
        self.turn_hint = turn
        if self.pos is not None and self.pos.turn != turn:
            self.pos.turn = turn
            self.pos.compute_key()
            self.seq += 1
            return True
        return False

    def set_orientation(self, orientation: str):
        self.turn_hint = WHITE if orientation == "white" else BLACK

    @property
    def placement(self) -> str | None:
        return self.pos.placement() if self.pos else None

    def update(self, placement: str, confidence: float = 0.0) -> bool:
        if not placement or len(placement.split("/")) != 8:
            return False
        if self.manual:
            return False
        self.confidence = float(confidence)

        if placement != self.pending[0]:
            self.pending = (placement, 1)
            return False
        self.pending = (placement, self.pending[1] + 1)
        if self.pending[1] < 2:
            return False
        if self.pos is not None and placement == self.pos.placement():
            return False

        if self.pos is None:
            # В стандартной начальной позиции первым всегда ходит белый
            start_turn = WHITE if placement == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR" else self.turn_hint
            self.pos = board_from_placement(placement, start_turn)
            self.last_move = None
        else:
            mv = infer_move(self.pos, placement)
            if mv is None and self.pending[1] < REBUILD_STABILITY:
                return False
            if mv is not None:
                self.pos.make_move(mv)
                self.last_move = mv
            else:
                moved = side_that_moved(self.pos.placement(), placement)
                if moved is not None:
                    nxt = BLACK if moved == WHITE else WHITE
                else:
                    nxt = BLACK if self.pos.turn == WHITE else WHITE
                self.pos = board_from_placement(placement, nxt)
                self.last_move = None
        self.seq += 1
        return True


def cell_center(rect, sq: int, orientation: str):
    x, y, w, h = rect
    col, row = square_to_cell(sq, orientation)
    cw, ch = w / 8.0, h / 8.0
    return (x + (col + 0.5) * cw, y + (row + 0.5) * ch, cw, ch)


def fmt_eval(res: dict, turn: int) -> str:
    mate = res.get("mate", 0)
    if mate:
        good = (mate > 0) == (turn == WHITE)
        return ("+" if good else "-") + f"мат в {abs(mate)}"
    score = int(res.get("score", 0))
    if turn == BLACK:
        score = -score
    if abs(score) >= 30000:
        return "+" if score > 0 else "-"
    return f"{score / 100.0:+.2f}"


def _enable_dpi() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _click_through(widget) -> None:
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetParent(widget.winfo_id()) or widget.winfo_id()
        GWL_EXSTYLE, W = -20, 0x80000 | 0x20 | 0x08000000 | 0x80
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | W)
        # Исключить оверлей из снимков экрана (WDA_EXCLUDEFROMCAPTURE):
        # Стрелки и сетка видны игроку на экране, но mss видит только чистую доску!
        user32.SetWindowDisplayAffinity(hwnd, 0x00000011)
    except Exception:
        pass


def _monitor_rect(monitor_idx: int = 1) -> tuple[int, int, int, int]:
    from .vision import _sct
    sct = _sct()
    idx = max(0, min(monitor_idx, len(sct.monitors) - 1))
    m = sct.monitors[idx]
    return int(m["left"]), int(m["top"]), int(m["width"]), int(m["height"])


def _to_ppm(img_bgr: np.ndarray) -> bytes:
    rgb = np.ascontiguousarray(img_bgr[:, :, ::-1])
    h, w = rgb.shape[:2]
    return f"P6 {w} {h} 255\n".encode("ascii") + rgb.tobytes()


class App:
    def __init__(self, root: tk.Tk, start_threads: bool = True):
        self.root = root
        self.cfg = load_config()
        self.lock = threading.Lock()
        self.stop_evt = threading.Event()
        self.enabled = bool(self.cfg.get("enabled", True))
        self.overlay_visible = True
        self.force_detect = False
        self.screen_rect = None
        self._auto_rect = None
        self._auto_until = 0.0
        self.confidence = 0.0
        self.last_played_seq = -1
        self.current_orientation = "white"
        self.locked_orientation = None

        self.tracker = Tracker(WHITE if self.cfg.get("turn_hint") == "white" else BLACK)
        self.engine = Engine()
        self.engine_stop = threading.Event()
        self.wake = threading.Event()
        self.result = None
        self.result_seq = -1
        self.result_turn = WHITE
        self.status = "запуск…"
        self._last_fen = ""

        from .voice import VoiceAnnouncer
        self.announcer = VoiceAnnouncer()
        self.engine.set_multipv(max(1, int(self.cfg.get("multipv", 2))))
        self.engine.set_elo(self.cfg.get("elo", 0))

        self.root.title("chess4-screen-bot")
        self.root.geometry("460x740+30+30")
        self.root.configure(bg=PANEL_BG)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_panel()
        self._build_overlay()

        if start_threads:
            threading.Thread(target=self._capture_loop, daemon=True).start()
            threading.Thread(target=self._engine_loop, daemon=True).start()
            threading.Thread(target=self._hotkey_loop, daemon=True).start()
        self.root.after(100, self._refresh)

    def _build_panel(self):
        st = ttk.Style(self.root)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure("TFrame", background=PANEL_BG)
        st.configure("TLabel", background=PANEL_BG, foreground="#f2f2f7", font=("Segoe UI", 10))
        st.configure("Hdr.TLabel", background=PANEL_BG, foreground="#ffffff", font=("Segoe UI", 13, "bold"))
        st.configure("Status.TLabel", background=PANEL_BG, foreground="#8ef58e", font=("Segoe UI", 9))
        st.configure("Big.TLabel", background=PANEL_BG, foreground="#ffd400", font=("Segoe UI", 28, "bold"))
        st.configure("Warn.TLabel", background=PANEL_BG, foreground="#ff453a", font=("Segoe UI", 9, "bold"))
        st.configure("TButton", font=("Segoe UI", 9), padding=3)

        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        hdr = ttk.Frame(main)
        hdr.pack(fill="x", pady=(0, 4))
        ttk.Label(hdr, text=f"chess4 (движок: {self.engine.name})", style="Hdr.TLabel").pack(side="left")
        self.lbl_status = ttk.Label(hdr, text=self.status, style="Status.TLabel")
        self.lbl_status.pack(side="right")

        card = ttk.Frame(main, padding=8)
        card.pack(fill="x", pady=(0, 6))
        self.lbl_move = ttk.Label(card, text="—", style="Big.TLabel")
        self.lbl_move.pack(anchor="w")

        info = ttk.Frame(card)
        info.pack(fill="x", pady=(2, 0))
        self.lbl_eval = ttk.Label(info, text="", font=("Segoe UI", 12, "bold"))
        self.lbl_eval.pack(side="right")
        self.lbl_depth = ttk.Label(info, text="", font=("Segoe UI", 9), foreground="#8e8e93")
        self.lbl_depth.pack(side="left")

        fen_box = ttk.Frame(main)
        fen_box.pack(fill="x", pady=(0, 4))
        ttk.Label(fen_box, text="Позиция (FEN):").pack(side="left")
        ttk.Button(fen_box, text="Копировать", width=11, command=self._copy_fen).pack(side="right")
        self.ent_fen = ttk.Entry(main, font=("Consolas", 9))
        self.ent_fen.pack(fill="x", pady=(2, 2))
        self.ent_fen.bind("<Return>", lambda _e: self._apply_fen())

        self.lbl_warn = ttk.Label(main, text="", style="Warn.TLabel")
        self.lbl_warn.pack(anchor="w", pady=(0, 4))

        grid = ttk.Frame(main)
        grid.pack(fill="x", pady=(0, 6))
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

        ttk.Label(grid, text="Ориентация:").grid(row=0, column=0, sticky="w")
        self.cmb_orient = ttk.Combobox(grid, values=["Авто", "Белые снизу", "Чёрные снизу"], state="readonly", width=14)
        orient_cfg = self.cfg.get("orientation", "auto")
        self.cmb_orient.set("Белые снизу" if orient_cfg == "white" else ("Чёрные снизу" if orient_cfg == "black" else "Авто"))
        self.cmb_orient.grid(row=1, column=0, sticky="we", padx=(0, 4))
        self.cmb_orient.bind("<<ComboboxSelected>>", self._on_orient)

        ttk.Label(grid, text="Мои фигуры:").grid(row=0, column=1, sticky="w")
        self.cmb_my = ttk.Combobox(grid, values=["Белые", "Чёрные", "Любые"], state="readonly", width=14)
        self.cmb_my.set({"white": "Белые", "black": "Чёрные", "any": "Любые"}.get(self.cfg.get("my_color", "white"), "Белые"))
        self.cmb_my.grid(row=1, column=1, sticky="we")
        self.cmb_my.bind("<<ComboboxSelected>>", self._on_my_color)

        ttk.Label(grid, text="Монитор:").grid(row=2, column=0, sticky="w", pady=(4, 0))
        from .vision import _sct
        mon_count = len(_sct().monitors) - 1
        mon_values = [f"Монитор {i}" for i in range(1, max(1, mon_count) + 1)]
        self.cmb_mon = ttk.Combobox(grid, values=mon_values, state="readonly", width=14)
        cur_mon = self.cfg.get("monitor", 1)
        self.cmb_mon.set(f"Монитор {cur_mon}" if cur_mon <= len(mon_values) else mon_values[0])
        self.cmb_mon.grid(row=3, column=0, sticky="we", padx=(0, 4))
        self.cmb_mon.bind("<<ComboboxSelected>>", self._on_monitor)

        ttk.Label(grid, text="Ход / Время:").grid(row=2, column=1, sticky="w", pady=(4, 0))
        turn_sub = ttk.Frame(grid)
        turn_sub.grid(row=3, column=1, sticky="we")
        self.btn_turn = ttk.Button(turn_sub, text="Белые" if self.cfg.get("turn_hint") == "white" else "Чёрные",
                                   command=self._on_turn, width=7)
        self.btn_turn.pack(side="left", fill="x", expand=True, padx=(0, 2))
        self.spn_think = ttk.Spinbox(turn_sub, from_=200, to=15000, increment=100, width=7)
        self.spn_think.delete(0, "end")
        self.spn_think.insert(0, str(int(self.cfg.get("think_ms", 1200))))
        self.spn_think.pack(side="right")

        ttk.Label(grid, text="Сила (Elo):").grid(row=4, column=0, sticky="w", pady=(4, 0))
        self.cmb_elo = ttk.Combobox(grid, values=["Максимум (Stockfish)", "Гроссмейстер (2500)", "Мастер (2100)", "КМС (1800)", "Любитель (1500)"], state="readonly", width=14)
        elo_map = {0: "Максимум (Stockfish)", 2500: "Гроссмейстер (2500)", 2100: "Мастер (2100)", 1800: "КМС (1800)", 1500: "Любитель (1500)"}
        self.cmb_elo.set(elo_map.get(self.cfg.get("elo", 0), "Максимум (Stockfish)"))
        self.cmb_elo.grid(row=5, column=0, sticky="we", padx=(0, 4))
        self.cmb_elo.bind("<<ComboboxSelected>>", self._on_elo)

        ttk.Label(grid, text="Линий (MultiPV):").grid(row=4, column=1, sticky="w", pady=(4, 0))
        self.cmb_mpv = ttk.Combobox(grid, values=["Без стрелок (только голос / окно)", "1 лучшая стрелка", "2 стрелки", "3 стрелки"], state="readonly", width=14)
        mpv_map = {0: "Без стрелок (только голос / окно)", 1: "1 лучшая стрелка", 2: "2 стрелки", 3: "3 стрелки"}
        self.cmb_mpv.set(mpv_map.get(self.cfg.get("multipv", 2), "2 стрелки"))
        self.cmb_mpv.grid(row=5, column=1, sticky="we")
        self.cmb_mpv.bind("<<ComboboxSelected>>", self._on_multipv)

        # Микро-подгонка рамки доски (Nudge buttons)
        nudge_box = ttk.LabelFrame(main, text="Точная подгонка сетки (±2 px):", padding=4)
        nudge_box.pack(fill="x", pady=(2, 6))
        n_btns = ttk.Frame(nudge_box)
        n_btns.pack(fill="x")
        ttk.Button(n_btns, text="◄ Влево", width=8, command=lambda: self._nudge(-2, 0, 0, 0)).pack(side="left", expand=True)
        ttk.Button(n_btns, text="▲ Вверх", width=8, command=lambda: self._nudge(0, -2, 0, 0)).pack(side="left", expand=True)
        ttk.Button(n_btns, text="▼ Вниз", width=8, command=lambda: self._nudge(0, 2, 0, 0)).pack(side="left", expand=True)
        ttk.Button(n_btns, text="► Вправо", width=8, command=lambda: self._nudge(2, 0, 0, 0)).pack(side="left", expand=True)
        ttk.Button(n_btns, text="— Уменьш", width=9, command=lambda: self._nudge(0, 0, -2, -2)).pack(side="left", expand=True)
        ttk.Button(n_btns, text="+ Увелич", width=9, command=lambda: self._nudge(0, 0, 2, 2)).pack(side="left", expand=True)

        btns = ttk.Frame(main)
        btns.pack(fill="x", pady=(2, 0))
        ttk.Button(btns, text="Найти доску", command=self._find_board).pack(side="left", fill="x", expand=True)
        ttk.Button(btns, text="Выделить область", command=self._select_area).pack(side="left", fill="x", expand=True, padx=4)
        self.btn_pause = ttk.Button(btns, text="Пауза" if self.enabled else "Пуск", command=self._toggle)
        self.btn_pause.pack(side="left", fill="x", expand=True)

        btns2 = ttk.Frame(main)
        btns2.pack(fill="x", pady=(4, 0))
        self.btn_grid = ttk.Button(btns2, text="Сетка: ВКЛ" if self.cfg.get("show_grid") else "Сетка: выкл",
                                   command=self._toggle_grid)
        self.btn_grid.pack(side="left", fill="x", expand=True)
        self.btn_auto = ttk.Button(btns2, text="Автоход: ВКЛ" if self.cfg.get("automove") else "Автоход: выкл",
                                   command=self._toggle_automove)
        self.btn_auto.pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(btns2, text="Сброс", command=self._reset).pack(side="left", fill="x", expand=True)

        btns3 = ttk.Frame(main)
        btns3.pack(fill="x", pady=(4, 0))
        self.btn_voice = ttk.Button(btns3, text="Озвучка: ВКЛ" if self.cfg.get("voice") else "Озвучка: выкл",
                                    command=self._toggle_voice)
        self.btn_voice.pack(side="left", fill="x", expand=True)
        self.btn_stealth = ttk.Button(btns3, text="Анти-чит: ВКЛ" if self.cfg.get("human_mouse", True) else "Анти-чит: выкл",
                                      command=self._toggle_stealth)
        self.btn_stealth.pack(side="left", fill="x", expand=True, padx=4)
        self.btn_drag = ttk.Button(btns3, text="Тянуть" if self.cfg.get("drag_drop", True) else "2 клика",
                                   command=self._toggle_drag)
        self.btn_drag.pack(side="left", fill="x", expand=True)

        hint_box = ttk.Frame(main)
        hint_box.pack(fill="x", pady=(5, 0))
        ttk.Label(hint_box, text="Горячие клавиши (ноутбук): Alt+1: Пуск | Alt+2: Сброс | Alt+3: Чей ход | Alt+4: Мои фигуры",
                  font=("Segoe UI", 8), foreground="#8e8e93").pack(anchor="center")
        ttk.Label(hint_box, text="Alt+5: Скрыть/Показать стрелки | Alt+6: Пересчёт | ~ : Пуск | Esc: Паника",
                  font=("Segoe UI", 8), foreground="#8e8e93").pack(anchor="center")

    def _build_overlay(self):
        left, top, sw, sh = _monitor_rect(self.cfg.get("monitor", 1))
        self.ov = tk.Toplevel(self.root)
        self.ov.withdraw()
        self.ov.overrideredirect(True)
        self.ov.attributes("-topmost", True)
        try:
            self.ov.attributes("-transparentcolor", TRANSPARENT)
        except tk.TclError:
            pass
        self.ov.geometry(f"{sw}x{sh}+{left}+{top}")
        self.ov_canvas = tk.Canvas(self.ov, bg=TRANSPARENT, highlightthickness=0, bd=0)
        self.ov_canvas.pack(fill="both", expand=True)
        self.ov.deiconify()
        self.ov.update_idletasks()
        _click_through(self.ov)

    # ------------------------------------------------------------- потоки
    def _capture_loop(self):
        while not self.stop_evt.is_set():
            if self.enabled:
                try:
                    self._tick()
                except Exception as exc:
                    self._set_status(f"ошибка захвата: {exc}")
            self.stop_evt.wait(max(60, int(self.cfg.get("poll_ms", 300))) / 1000.0)

    def _tick(self):
        now = time.monotonic()
        rect_cfg = self.cfg.get("rect")
        mon_idx = self.cfg.get("monitor", 1)
        if self.force_detect:
            self.force_detect = False
            rect_cfg = None
            self._auto_rect = None

        if rect_cfg:
            left, top, w, h = rect_cfg
            img = capture({"left": left, "top": top, "width": w, "height": h}, monitor_idx=mon_idx)
            rect = (0, 0, w, h)
            self.screen_rect = (left, top, w, h)
        else:
            img = capture(None, monitor_idx=mon_idx)
            rect = self._auto_rect
            if rect is None or now >= self._auto_until:
                rect = detect_board(img)
                if rect is None:
                    self.screen_rect = None
                    self._auto_rect = None
                    self._set_status("доска не найдена — «Выделите область»")
                    return
                self._auto_rect = rect
                self._auto_until = now + REDETECT_S
            m_left, m_top, _, _ = _monitor_rect(mon_idx)
            self.screen_rect = (int(m_left + rect[0]), int(m_top + rect[1]),
                                int(rect[2]), int(rect[3]))

        orient_cfg = self.cfg.get("orientation", "auto")
        my_color = self.cfg.get("my_color", "white")

        if orient_cfg in ("white", "black"):
            effective_orient = orient_cfg
        else:
            # В режиме "Авто": если ориентация уже зафиксирована для текущей партии,
            # не позволяем ходам в эндшпиле случайно перевернуть доску!
            if self.tracker.pos is not None and getattr(self, "locked_orientation", None):
                effective_orient = self.locked_orientation
            else:
                effective_orient = "auto"

        res = read_board(img, rect, effective_orient, my_color_hint=my_color)
        placement = res.get("placement")
        if not placement:
            self._set_status("позиция не распознана")
            return
        self.confidence = float(res.get("confidence", 0.0))
        self.current_orientation = res.get("orientation", "white")
        if self.tracker.pos is not None and not getattr(self, "locked_orientation", None):
            self.locked_orientation = self.current_orientation

        if self.tracker.update(placement, self.confidence):
            self.engine_stop.set()
            self.wake.set()
            self._set_status(f"позиция распознана ({self.confidence:.2f})")

    def _calc_think_time(self, pos: Position) -> int:
        base_ms = int(self.cfg.get("think_ms", 1200))
        if not self.cfg.get("human_time", True):
            return base_ms

        if pos.full <= 4:
            return int(random.uniform(280, 520))

        legal = pos.legal_moves()
        if len(legal) <= 2:
            return int(random.uniform(280, 520))

        factor = random.uniform(0.80, 1.30)
        return max(300, int(base_ms * factor))

    def _engine_loop(self):
        while not self.stop_evt.is_set():
            if not self.wake.wait(0.2):
                continue
            self.wake.clear()
            with self.lock:
                pos = self.tracker.pos.clone() if self.tracker.pos else None
                seq = self.tracker.seq
                placement = self.tracker.placement
                conf = self.tracker.confidence
                manual = self.tracker.manual
            if pos is None:
                continue
            if not self.enabled:
                self._set_status("пауза")
                continue

            # Защита от мусора: строгая проверка валидности расстановки
            if not manual:
                issues = placement_issues(placement)
                if issues:
                    self._clear_result()
                    self._set_status("некорректная позиция: " + issues[0])
                    continue
                if conf < CONF_MIN:
                    self._clear_result()
                    self._set_status(f"уверенность {conf:.2f} — позиция неясна")
                    continue

            if not self._my_turn(pos):
                self._clear_result()
                self._set_status("жду ход соперника")
                continue

            self.engine_stop.clear()
            self._set_status("думаю…")
            think_ms = self._calc_think_time(pos)
            try:
                res = self.engine.analyze(pos, think_ms, self.engine_stop)
            except Exception as exc:
                self._set_status(f"ошибка движка: {exc}")
                continue

            if self.stop_evt.is_set():
                return
            if seq != self.tracker.seq:
                continue
            with self.lock:
                self.result = res
                self.result_seq = seq
                self.result_turn = pos.turn

            if res.get("move"):
                self._set_status("готово")
                if self.cfg.get("voice", False):
                    san = res.get("san")
                    if san:
                        self.announcer.speak_move(san)
                if self.cfg.get("automove", False) and seq != self.last_played_seq:
                    self.last_played_seq = seq
                    threading.Thread(target=self._execute_automove, args=(res["move"], pos), daemon=True).start()
            else:
                self._set_status("хода нет")

    def _execute_automove(self, move, pos):
        """Человекоподобное выполнение хода (Drag-and-Drop или 2 клика)."""
        try:
            time.sleep(random.uniform(0.08, 0.20))
            rect = self.screen_rect
            if not rect:
                return
            orient = getattr(self, "current_orientation", self.cfg.get("orientation", "white"))
            if orient == "auto":
                orient = "black" if self.cfg.get("my_color") == "black" else "white"
            fx, fy, _, _ = cell_center(rect, move[0], orient)
            tx, ty, _, _ = cell_center(rect, move[1], orient)

            use_drag = self.cfg.get("drag_drop", True)
            if self.cfg.get("human_mouse", True):
                if use_drag:
                    from .mouse import human_drag_and_drop
                    human_drag_and_drop(fx, fy, tx, ty)
                else:
                    from .mouse import human_move_and_click
                    human_move_and_click(fx, fy, tx, ty)
            else:
                import pyautogui
                pyautogui.FAILSAFE = False
                if use_drag:
                    pyautogui.moveTo(fx, fy, duration=0.10)
                    pyautogui.mouseDown()
                    time.sleep(0.06)
                    pyautogui.moveTo(tx, ty, duration=0.12)
                    time.sleep(0.06)
                    pyautogui.mouseUp()
                else:
                    pyautogui.moveTo(fx, fy, duration=0.12)
                    pyautogui.click()
                    time.sleep(0.10)
                    pyautogui.moveTo(tx, ty, duration=0.12)
                    pyautogui.click()
        except Exception as e:
            print("Ошибка автохода:", e)

    def _hotkey_loop(self):
        """Фоновый перехват системных горячих клавиш (оптимизировано под ноутбуки без клавиш F)."""
        user32 = ctypes.windll.user32
        VK_MENU = 0x12   # Alt
        VK_ESC = 0x1B    # Esc
        VK_TILDE = 0xC0  # ~ ` Ё
        VK_F1, VK_F2, VK_F3, VK_F4, VK_F5, VK_F6, VK_F7 = (
            0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76
        )

        # Сочетания клавиш: (требуется_Alt, виртуальный_код)
        # На ноутбуке клавиши Alt+1..6 нажимаются одной левой рукой без клавиши Fn!
        combos = [
            ("PAUSE", [(True, 0x31), (True, 0x5A), (True, 0x50), (False, VK_TILDE), (False, VK_F4), (False, VK_F1)], self._toggle),
            ("RESET", [(True, 0x32), (True, 0x58), (True, 0x52), (False, VK_F5)], self._reset),
            ("TURN", [(True, 0x33), (True, 0x43), (True, 0x54), (False, VK_F6)], self._on_turn),
            ("MY_COLOR", [(True, 0x34), (True, 0x56), (True, 0x4D), (False, VK_F7)], self._toggle_my_color),
            ("OVERLAY", [(True, 0x35), (True, 0x42), (True, 0x4F), (False, VK_F2)], self._toggle_overlay),
            ("RECALC", [(True, 0x36), (True, 0x53), (False, VK_F3)], self._recalc),
            ("PANIC", [(False, VK_ESC)], self._panic_hide),
        ]

        pressed = set()
        while not self.stop_evt.is_set():
            alt_down = bool(user32.GetAsyncKeyState(VK_MENU) & 0x8000)
            for name, triggers, action in combos:
                is_triggered = False
                for req_alt, code in triggers:
                    key_down = bool(user32.GetAsyncKeyState(code) & 0x8000)
                    if key_down and (not req_alt or alt_down):
                        is_triggered = True
                        break

                if is_triggered and name not in pressed:
                    pressed.add(name)
                    self.root.after(0, action)
                elif not is_triggered and name in pressed:
                    pressed.remove(name)
            time.sleep(0.05)

    def _clear_result(self):
        with self.lock:
            self.result = None
            self.result_seq = -1

    def _my_turn(self, pos: Position) -> bool:
        my = self.cfg.get("my_color", "white")
        if my == "any":
            return True
        want = WHITE if my == "white" else BLACK
        return pos.turn == want

    def _on_close(self):
        self.stop_evt.set()
        self.wake.set()
        self.engine_stop.set()
        self.engine.close()
        try:
            self.ov.destroy()
        except Exception:
            pass
        self.root.destroy()

    def _set_status(self, text: str):
        self.status = text

    def _copy_fen(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.ent_fen.get())

    def _apply_fen(self):
        text = self.ent_fen.get().strip()
        if not text:
            return
        try:
            pos = Position.from_fen(text)
        except Exception as exc:
            self._set_status(f"FEN не разобран: {exc}")
            return
        issues = placement_issues(pos.placement())
        if issues:
            self._set_status("FEN невозможен: " + issues[0])
            return
        self.tracker.set_manual(pos)
        self.engine_stop.set()
        self.wake.set()
        self._set_status("позиция задана вручную")

    def _warn_text(self) -> str:
        if self.tracker.manual:
            return "позиция вручную"
        if self.tracker.pos is None:
            return ""
        issues = placement_issues(self.tracker.placement)
        if issues:
            return "⚠ " + issues[0]
        conf = self.tracker.confidence
        if conf < CONF_WARN:
            return f"уверенность {conf:.2f} — проверьте границы"
        return f"уверенность {conf:.2f}"

    def _find_board(self):
        self.cfg["rect"] = None
        self.force_detect = True
        self.tracker.reset()
        self._set_status("ищу доску…")

    def _toggle(self):
        self.enabled = not self.enabled
        self.cfg["enabled"] = self.enabled
        self.btn_pause.configure(text="Пауза" if self.enabled else "Пуск")
        if self.enabled:
            self.wake.set()

    def _toggle_grid(self):
        show = not self.cfg.get("show_grid", False)
        self.cfg["show_grid"] = show
        self.btn_grid.configure(text="Сетка: ВКЛ" if show else "Сетка: выкл")
        save_config(self.cfg)

    def _toggle_automove(self):
        auto = not self.cfg.get("automove", False)
        self.cfg["automove"] = auto
        self.btn_auto.configure(text="Автоход: ВКЛ" if auto else "Автоход: выкл")
        save_config(self.cfg)

    def _toggle_overlay(self):
        self.overlay_visible = not self.overlay_visible
        if not self.overlay_visible:
            self.ov_canvas.delete("all")

    def _panic_hide(self):
        self.overlay_visible = False
        self.ov_canvas.delete("all")

    def _recalc(self):
        self.wake.set()

    def _reset(self):
        self.tracker.reset()
        self.result = None
        self.result_seq = -1
        self.locked_orientation = None
        if getattr(self, "ov_canvas", None):
            self.ov_canvas.delete("all")
        self._set_status("позиция сброшена")

    def _nudge(self, dx: int, dy: int, dw: int, dh: int):
        """Сдвиг/изменение размера области доски с клавиатуры/кнопок."""
        rect = self.cfg.get("rect")
        if not rect:
            if self.screen_rect:
                rect = list(self.screen_rect)
            else:
                self._set_status("сначала выделите область")
                return
        x, y, w, h = rect
        new_rect = [max(0, x + dx), max(0, y + dy), max(32, w + dw), max(32, h + dh)]
        self.cfg["rect"] = new_rect
        self.screen_rect = tuple(new_rect)
        self.tracker.reset()
        self.locked_orientation = None
        save_config(self.cfg)
        self._set_status(f"рамка: {new_rect[0]},{new_rect[1]} ({new_rect[2]}x{new_rect[3]})")

    def _on_orient(self, _evt=None):
        text = self.cmb_orient.get()
        if text.startswith("Белые"):
            self.cfg["orientation"] = "white"
        elif text.startswith("Чёрные"):
            self.cfg["orientation"] = "black"
        else:
            self.cfg["orientation"] = "auto"
        self.locked_orientation = None
        self.tracker.reset()
        save_config(self.cfg)

    def _on_my_color(self, _evt=None):
        text = self.cmb_my.get()
        self.cfg["my_color"] = {"Белые": "white", "Чёрные": "black", "Любые": "any"}.get(text, "white")
        self.locked_orientation = None
        self.tracker.reset()
        save_config(self.cfg)
        self.wake.set()

    def _toggle_my_color(self):
        """Горячая клавиша F7: мгновенное переключение стороны (Белые <-> Чёрные)."""
        cur = self.cfg.get("my_color", "white")
        new_color = "black" if cur == "white" else "white"
        self.cfg["my_color"] = new_color
        color_text = "Белые" if new_color == "white" else "Чёрные"
        self.cmb_my.set(color_text)
        self.locked_orientation = None
        self.tracker.reset()
        save_config(self.cfg)
        self.wake.set()
        self._set_status(f"мои фигуры: {color_text.lower()}")

    def _on_monitor(self, _evt=None):
        text = self.cmb_mon.get()
        try:
            idx = int(text.split()[-1])
            self.cfg["monitor"] = idx
            save_config(self.cfg)
            left, top, sw, sh = _monitor_rect(idx)
            self.ov.geometry(f"{sw}x{sh}+{left}+{top}")
            self.tracker.reset()
            self._set_status(f"переключено на монитор {idx}")
        except Exception:
            pass

    def _on_turn(self):
        """Переключение очереди хода (Белые <-> Чёрные), кнопка и горячая клавиша F6."""
        new = BLACK if self.btn_turn["text"] == "Белые" else WHITE
        self.btn_turn.configure(text="Чёрные" if new == BLACK else "Белые")
        self.cfg["turn_hint"] = "white" if new == WHITE else "black"
        if self.tracker.set_turn(new):
            self.engine_stop.set()
            self.wake.set()
        save_config(self.cfg)
        self._set_status(f"ход переключён: {'белые' if new == WHITE else 'чёрные'}")

    def _on_elo(self, _evt=None):
        text = self.cmb_elo.get()
        mapping = {
            "Максимум (Stockfish)": 0,
            "Гроссмейстер (2500)": 2500,
            "Мастер (2100)": 2100,
            "КМС (1800)": 1800,
            "Любитель (1500)": 1500,
        }
        elo_val = mapping.get(text, 0)
        self.cfg["elo"] = elo_val
        self.engine.set_elo(elo_val)
        save_config(self.cfg)
        self.wake.set()

    def _on_multipv(self, _evt=None):
        text = self.cmb_mpv.get()
        mapping = {
            "Без стрелок (только голос / окно)": 0,
            "1 лучшая стрелка": 1,
            "2 стрелки": 2,
            "3 стрелки": 3,
        }
        val = mapping.get(text, 2)
        self.cfg["multipv"] = val
        self.engine.set_multipv(max(1, val))
        save_config(self.cfg)
        if val == 0:
            if getattr(self, "ov_canvas", None):
                self.ov_canvas.delete("all")
            self._set_status("режим без стрелок (скрытно / только озвучка)")
        else:
            self._set_status(f"линий анализа: {val}")
        self.wake.set()

    def _toggle_voice(self):
        v = not self.cfg.get("voice", False)
        self.cfg["voice"] = v
        self.btn_voice.configure(text="Озвучка: ВКЛ" if v else "Озвучка: выкл")
        save_config(self.cfg)

    def _toggle_stealth(self):
        s = not self.cfg.get("human_mouse", True)
        self.cfg["human_mouse"] = s
        self.cfg["human_time"] = s
        self.btn_stealth.configure(text="Анти-чит: ВКЛ" if s else "Анти-чит: выкл")
        save_config(self.cfg)

    def _toggle_drag(self):
        d = not self.cfg.get("drag_drop", True)
        self.cfg["drag_drop"] = d
        self.btn_drag.configure(text="Тянуть" if d else "2 клика")
        save_config(self.cfg)
        self._set_status("автоход: перетаскивание (drag)" if d else "автоход: 2 клика")

    def _select_area(self):
        """Интерактивное выделение области: строгое сохранение пользовательских координат."""
        mon_idx = self.cfg.get("monitor", 1)
        left_m, top_m, sw, sh = _monitor_rect(mon_idx)
        img = capture(None, monitor_idx=mon_idx)
        fd, path = tempfile.mkstemp(suffix=".ppm")
        try:
            os.write(fd, _to_ppm(img))
        finally:
            os.close(fd)

        sel = tk.Toplevel(self.root)
        sel.overrideredirect(True)
        sel.attributes("-topmost", True)
        sel.geometry(f"{sw}x{sh}+{left_m}+{top_m}")
        canvas = tk.Canvas(sel, highlightthickness=0, cursor="cross")
        photo = tk.PhotoImage(file=path)
        canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas.pack(fill="both", expand=True)
        sel._photo = photo
        state = {"start": None}

        def press(ev):
            state["start"] = (ev.x, ev.y)

        def drag(ev):
            if not state["start"]:
                return
            canvas.delete("sel")
            canvas.create_rectangle(state["start"][0], state["start"][1],
                                    ev.x, ev.y, outline=ARROW, width=3, tags="sel")

        def release(ev):
            start = state["start"]
            sel.destroy()
            try:
                os.unlink(path)
            except OSError:
                pass
            if not start:
                return
            x0, y0 = start
            x1, y1 = ev.x, ev.y
            left, top = min(x0, x1), min(y0, y1)
            ww, hh = abs(x1 - x0), abs(y1 - y0)
            if ww < 48 or hh < 48:
                self._set_status("область слишком мала")
                return

            # Автоматическая привязка к клеткам доски (если сетка найдена в пределах окрестности)
            snapped = snap_selection(img, (int(left), int(top), int(ww), int(hh)))
            s_left, s_top, s_ww, s_hh = snapped

            abs_left = left_m + int(s_left)
            abs_top = top_m + int(s_top)
            self.cfg["rect"] = [abs_left, abs_top, int(s_ww), int(s_hh)]
            self.force_detect = False
            self._auto_rect = None
            self.tracker.reset()
            self.locked_orientation = None
            save_config(self.cfg)
            if (s_left, s_top, s_ww, s_hh) != (int(left), int(top), int(ww), int(hh)):
                self._set_status(f"область авто-выровнена: {s_ww}x{s_hh}")
            else:
                self._set_status(f"область сохранена: {s_ww}x{s_hh}")

        canvas.bind("<ButtonPress-1>", press)
        canvas.bind("<B1-Motion>", drag)
        canvas.bind("<ButtonRelease-1>", release)

    def _refresh(self):
        try:
            if not self.root.winfo_exists():
                return
        except tk.TclError:
            return
        self.lbl_status.configure(text=self.status)

        pos = self.tracker.pos
        fen = pos.to_fen() if pos is not None else ""
        if fen != self._last_fen:
            self._last_fen = fen
            self.ent_fen.delete(0, "end")
            self.ent_fen.insert(0, fen)
        self.lbl_warn.configure(text=self._warn_text())
        if pos is not None:
            turn_str = "Чёрные" if pos.turn == BLACK else "Белые"
            if self.btn_turn["text"] != turn_str:
                self.btn_turn.configure(text=turn_str)

        res = self.result
        fresh = res is not None and self.result_seq == self.tracker.seq
        if fresh and res.get("move") is not None:
            lines = res.get("lines") or []
            if len(lines) > 1 and int(self.cfg.get("multipv", 2)) > 1:
                display_parts = []
                for li in lines[:int(self.cfg.get("multipv", 2))]:
                    san_str = li.get("san") or ""
                    sc_str = fmt_eval({"score": li.get("score", 0), "mate": 0}, self.result_turn)
                    display_parts.append(f"{san_str} ({sc_str})")
                self.lbl_move.configure(text="   |   ".join(display_parts))
            else:
                san = res.get("san") or ""
                self.lbl_move.configure(text=san + ("  📖" if res.get("book") else ""))
            self.lbl_eval.configure(text=fmt_eval(res, self.result_turn))
            self.lbl_depth.configure(text=f"глубина {res.get('depth', 0)} · {res.get('nps', 0)} уз/с")
        elif res is not None and res.get("move") is None and fresh:
            self.lbl_move.configure(text="—")
            self.lbl_eval.configure(text="мат / пат")
            self.lbl_depth.configure(text="")
        elif not self.enabled:
            self.lbl_move.configure(text="⏸")
            self.lbl_eval.configure(text="")
            self.lbl_depth.configure(text="пауза")
        else:
            busy = self.status.startswith("думаю")
            self.lbl_move.configure(text="…" if busy else "—")
            self.lbl_eval.configure(text="")
            self.lbl_depth.configure(text="расчёт" if busy else self.status)

        self._draw_overlay(fresh, res)
        self.root.after(100, self._refresh)

    def _draw_overlay(self, fresh: bool, res):
        cv = self.ov_canvas
        cv.delete("all")
        if not self.overlay_visible:
            return
        rect = self.screen_rect
        if rect is None:
            return

        x, y, w, h = rect
        mon_idx = self.cfg.get("monitor", 1)
        mon_left, mon_top, _, _ = _monitor_rect(mon_idx)
        # Относительные координаты внутри холста оверлея текущего монитора
        rx = x - mon_left
        ry = y - mon_top
        cw, ch = w / 8.0, h / 8.0

        # Отрисовка проверочной сетки
        if self.cfg.get("show_grid", False):
            for i in range(9):
                cv.create_line(rx + i * cw, ry, rx + i * cw, ry + h, fill=GRID_COLOR, width=1)
                cv.create_line(rx, ry + i * ch, rx + w, ry + i * ch, fill=GRID_COLOR, width=1)

        # Отрисовка стрелок ходов (MultiPV)
        mpv_count = int(self.cfg.get("multipv", 2))
        if mpv_count == 0:
            return  # Режим без стрелок: оверлей остаётся чистым (только озвучка / панель)

        if not fresh or res is None:
            return
        m = res.get("move")
        if m is None:
            return
        orientation = getattr(self, "current_orientation", self.cfg.get("orientation", "white"))
        if orientation == "auto":
            orientation = "black" if self.cfg.get("my_color") == "black" else "white"

        lines = res.get("lines") or []
        if not lines:
            lines = [{"rank": 1, "move": m, "san": res.get("san"), "score": res.get("score")}]

        arrow_palette = [ARROW, ARROW_SECONDARY, ARROW_TERTIARY]
        arrow_widths = [6, 4, 3]

        for idx, line_info in enumerate(lines[:int(self.cfg.get("multipv", 2))]):
            mv = line_info.get("move")
            if not mv:
                continue
            color = arrow_palette[min(idx, len(arrow_palette) - 1)]
            width = arrow_widths[min(idx, len(arrow_widths) - 1)]

            if idx == 0:
                for sq, hl_col in ((mv[0], HL_FROM), (mv[1], HL_TO)):
                    col, row = square_to_cell(sq, orientation)
                    cv.create_rectangle(rx + col * cw + 2, ry + row * ch + 2,
                                        rx + (col + 1) * cw - 2, ry + (row + 1) * ch - 2,
                                        outline=hl_col, width=3)

            x1, y1, _, _ = cell_center((rx, ry, w, h), mv[0], orientation)
            x2, y2, _, _ = cell_center((rx, ry, w, h), mv[1], orientation)

            ashape = (16, 20, 7) if idx == 0 else (12, 16, 5)
            cv.create_line(x1, y1, x2, y2, fill=color, width=width, arrow="last",
                           arrowshape=ashape, capstyle="round", joinstyle="round")
            rad = 6 if idx == 0 else 4
            cv.create_oval(x1 - rad, y1 - rad, x1 + rad, y1 + rad, fill=color, outline="")
