"""Шахматный движок: оценка + поиск (iterative deepening, alpha-beta, TT).

Если рядом лежит stockfish.exe — используется он (UCI), иначе встроенный
собственный поиск. Оба отдают одинаковый результат analyze() -> dict.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time

from .board import (BISHOP, BLACK, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE,
                    COLOR_MASK, FL_EP, Position)

INF = 10 ** 6
MATE = 30000
MATE_THRESHOLD = MATE - 1000

EXACT, LOWER, UPPER = 0, 1, 2

MATERIAL = {PAWN: 100, KNIGHT: 325, BISHOP: 335, ROOK: 500, QUEEN: 975, KING: 0}
MATERIAL_LIST = [0, 100, 325, 335, 500, 975, 0]     # индекс = тип фигуры
PHASE_VALUE = {PAWN: 0, KNIGHT: 1, BISHOP: 1, ROOK: 2, QUEEN: 4, KING: 0}
PHASE_LIST = [0, 0, 1, 1, 2, 4, 0]
MAX_PHASE = 24

# --- таблицы полей (индекс 0 = a8, запись от имени белых) -------------------
_PST_PAWN_MID = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]
_PST_PAWN_END = [
     0,  0,  0,  0,  0,  0,  0,  0,
    80, 80, 80, 80, 80, 80, 80, 80,
    50, 50, 50, 50, 50, 50, 50, 50,
    30, 30, 30, 30, 30, 30, 30, 30,
    20, 20, 20, 20, 20, 20, 20, 20,
    10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10,
     0,  0,  0,  0,  0,  0,  0,  0,
]
_PST_KNIGHT = [
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50,
]
_PST_BISHOP = [
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -20,-10,-10,-10,-10,-10,-10,-20,
]
_PST_ROOK = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]
_PST_QUEEN = [
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20,
]
_PST_KING_MID = [
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20,
]
_PST_KING_END = [
   -50,-40,-30,-20,-20,-30,-40,-50,
   -30,-20,-10,  0,  0,-10,-20,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-30,  0,  0,  0,  0,-30,-30,
   -50,-30,-30,-30,-30,-30,-30,-50,
]


def _build_pst():
    """PST для белых и чёрных, индекс = 0x88-квадрат (0..127 по игровым)."""
    out = {}
    for piece, base, eg in ((PAWN, _PST_PAWN_MID, _PST_PAWN_END),
                            (KNIGHT, _PST_KNIGHT, _PST_KNIGHT),
                            (BISHOP, _PST_BISHOP, _PST_BISHOP),
                            (ROOK, _PST_ROOK, _PST_ROOK),
                            (QUEEN, _PST_QUEEN, _PST_QUEEN),
                            (KING, _PST_KING_MID, _PST_KING_END)):
        w_mid, w_end = [0] * 128, [0] * 128
        b_mid, b_end = [0] * 128, [0] * 128
        for sq in range(128):
            if sq & 0x88:
                continue
            r, f = sq >> 4, sq & 7
            w_mid[sq] = base[(7 - r) * 8 + f]
            w_end[sq] = eg[(7 - r) * 8 + f]
            b_mid[sq] = base[r * 8 + f]
            b_end[sq] = eg[r * 8 + f]
        out[piece] = (w_mid, w_end, b_mid, b_end)
    return out


PST = _build_pst()
# PST по типу фигуры: [тип] -> (w_mid, w_end, b_mid, b_end)
PST_LIST = [None] + [PST[t] for t in (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)]

PASSED_PAWN = [0, 8, 12, 22, 40, 68, 110, 0]   # по рангу продвижения
DOUBLED_PAWN = -14
ISOLATED_PAWN = -14
BISHOP_PAIR = 35
ROOK_OPEN_FILE = 22
ROOK_SEMI_OPEN = 11
KING_SHIELD = 9
TEMPO = 12

_FILE_MASK = [0] * 128          # битовая маска горизонтали для каждого поля
for _sq in range(128):
    if not (_sq & 0x88):
        _FILE_MASK[_sq] = 1 << (_sq & 7)


def evaluate(pos: Position) -> int:
    """Оценка в сантипешках с точки зрения стороны, чей ход (с темпом)."""
    b = pos.board
    mg = eg = 0
    phase = 0
    pf = [[0] * 8, [0] * 8]                 # пешки по горизонталям
    pr = [[[] for _ in range(8)], [[] for _ in range(8)]]
    bishops = [0, 0]
    rooks = [[], []]

    for sq in range(128):
        if sq & 0x88:
            continue
        p = b[sq]
        if not p:
            continue
        ci = ((p & COLOR_MASK) >> 3) - 1     # 0 = белые, 1 = чёрные
        t = p & 7
        if t == PAWN:
            f = sq & 7
            pf[ci][f] += 1
            pr[ci][f].append(sq >> 4)
        elif t == BISHOP:
            bishops[ci] += 1
        elif t == ROOK:
            rooks[ci].append(sq)
        v = MATERIAL_LIST[t]
        phase += PHASE_LIST[t]
        w = PST_LIST[t]
        if ci == 0:
            mg += v + w[0][sq]
            eg += v + w[1][sq]
        else:
            mg -= v + w[2][sq]
            eg -= v + w[3][sq]

    # пешечная структура
    for ci in (0, 1):
        sign = 1 if ci == 0 else -1
        files, ranks = pf[ci], pr[ci]
        enemy_ranks = pr[1 - ci]
        for f in range(8):
            cnt = files[f]
            if not cnt:
                continue
            if cnt > 1:
                mg += sign * DOUBLED_PAWN * (cnt - 1)
                eg += sign * DOUBLED_PAWN * (cnt - 1)
            left = files[f - 1] if f > 0 else 0
            right = files[f + 1] if f < 7 else 0
            if not left and not right:
                mg += sign * ISOLATED_PAWN * cnt
                eg += sign * ISOLATED_PAWN * cnt
            for r in ranks[f]:
                passed = True
                for df in (f - 1, f, f + 1):
                    if df < 0 or df > 7:
                        continue
                    for er in enemy_ranks[df]:
                        if (er > r) if ci == 0 else (er < r):
                            passed = False
                            break
                    if not passed:
                        break
                if passed:
                    idx = r if ci == 0 else 7 - r
                    val = PASSED_PAWN[min(max(idx, 0), 7)]
                    mg += sign * val
                    eg += sign * int(val * 1.4)

    if bishops[0] >= 2:
        mg += BISHOP_PAIR
        eg += BISHOP_PAIR + 15
    if bishops[1] >= 2:
        mg -= BISHOP_PAIR
        eg -= BISHOP_PAIR + 15

    # ладьи на открытых линиях
    for ci in (0, 1):
        sign = 1 if ci == 0 else -1
        own_files, opp_files = pf[ci], pf[1 - ci]
        for sq in rooks[ci]:
            f = sq & 7
            if not own_files[f] and not opp_files[f]:
                mg += sign * ROOK_OPEN_FILE
                eg += sign * ROOK_OPEN_FILE
            elif not own_files[f]:
                mg += sign * ROOK_SEMI_OPEN
                eg += sign * ROOK_SEMI_OPEN

    # пешечный щит короля (только миттельшпиль)
    mg += _king_shield(pos, WHITE) - _king_shield(pos, BLACK)

    if phase > MAX_PHASE:
        phase = MAX_PHASE
    score = (mg * phase + eg * (MAX_PHASE - phase)) // MAX_PHASE
    if pos.turn == BLACK:
        score = -score
    return score + TEMPO


def _king_shield(pos: Position, color: int) -> int:
    b = pos.board
    ksq = pos.kings[color]
    if ksq < 0:
        return 0
    if not (0 <= ksq <= 127) or (ksq & 0x88):
        return 0
    f = ksq & 7
    r = ksq >> 4
    fwd = 1 if color == WHITE else -1
    score = 0
    for df in (-1, 0, 1):
        ff = f + df
        if ff < 0 or ff > 7:
            continue
        rr = r + fwd
        if 0 <= rr <= 7:
            sq = rr * 16 + ff
            p = b[sq]
            if p and (p & COLOR_MASK) == color and (p & 7) == PAWN:
                score += KING_SHIELD
    return score


# ==================================================================== поиск
class Searcher:
    MAX_PLY = 64

    def __init__(self, name="builtin"):
        self.name = name
        self.tt = {}
        self.killers = [[None, None] for _ in range(self.MAX_PLY + 2)]
        self.history = {}
        self.nodes = 0
        self.stopped = False
        self.deadline = 0.0
        self.stop_event = threading.Event()   # внутреннее прерывание поиска
        self.external_stop = None             # внешний Event (смена позиции)
        self._pv = [[] for _ in range(self.MAX_PLY + 2)]
        self.check_every = 2048

    # ---------------------------------------------------------- вспомогательное
    def _time_up(self):
        if self.stopped:
            return True
        if self.stop_event.is_set():
            self.stopped = True
            return True
        if self.external_stop is not None and self.external_stop.is_set():
            self.stopped = True
            return True
        if self.deadline and (self.nodes & 1023) == 0 and time.time() > self.deadline:
            self.stopped = True
            return True
        return False

    @staticmethod
    def _non_pawn_material(pos: Position, color: int) -> int:
        total = 0
        b = pos.board
        for sq in range(128):
            if sq & 0x88:
                continue
            p = b[sq]
            if p and (p & COLOR_MASK) == color and (p & 7) != PAWN:
                total += 1
        return total

    def _score_moves(self, pos, moves, tt_move, ply):
        board = pos.board
        killers = self.killers[ply] if ply < len(self.killers) else (None, None)
        scored = []
        for m in moves:
            frm, to, promo, flag = m
            s = 0
            if tt_move is not None and m == tt_move:
                s = 10 ** 7
            else:
                victim = board[to]
                if victim:
                    vt = victim & 7
                    attacker = board[frm] & 7
                    s = 10 ** 6 + 100 * MATERIAL[vt] - 10 * MATERIAL[attacker]
                elif flag == FL_EP:
                    s = 10 ** 6 + 100 * MATERIAL[PAWN] - 10 * MATERIAL[PAWN]
                elif promo == QUEEN:
                    s = 9 * 10 ** 5
                elif promo:
                    s = 5 * 10 ** 5
                else:
                    if m == killers[0]:
                        s = 8 * 10 ** 5
                    elif m == killers[1]:
                        s = 7 * 10 ** 5
                    else:
                        s = self.history.get((frm, to), 0)
            scored.append((s, m))
        scored.sort(key=lambda x: -x[0])
        return [m for _, m in scored]

    # ------------------------------------------------------------------ корень
    def search(self, pos: Position, movetime_ms: int = 1500,
               max_depth: int = 64, max_nodes: int = 0) -> dict:
        self.nodes = 0
        self.stopped = False
        self.tt = {}
        self.killers = [[None, None] for _ in range(self.MAX_PLY + 2)]
        self.history = {}
        self.deadline = time.time() + max(0.01, movetime_ms / 1000.0)
        self.stop_event.clear()

        t0 = time.time()
        legal = pos.legal_moves()
        if not legal:
            return {"move": None, "score": -MATE if pos.in_check() else 0,
                    "depth": 0, "nodes": 0, "time": 0.0, "pv": [],
                    "mate": -1 if pos.in_check() else 0}

        best_move = legal[0]
        best_score = 0
        completed_depth = 0
        pv = [best_move]
        alpha, beta = -INF, INF

        for depth in range(1, max_depth + 1):
            self._pv[0] = []
            score, move, line = self._root(pos, depth, alpha, beta)
            if self.stopped and depth > 1:
                break
            if move is not None:
                best_move, best_score, pv = move, score, line or [move]
                completed_depth = depth
            if time.time() > self.deadline:
                break
            if max_nodes and self.nodes >= max_nodes:
                break
            # асимметричное окно вокруг найденного хода
            alpha, beta = -INF, INF

        elapsed = max(1e-6, time.time() - t0)
        mate = 0
        if abs(best_score) >= MATE_THRESHOLD:
            n = MATE - abs(best_score)
            mate = (1 if best_score > 0 else -1) * max(1, (n + 1) // 2)
        return {
            "move": best_move,
            "uci": pos.uci(best_move) if best_move else None,
            "san": pos.san(best_move) if best_move else None,
            "score": best_score,
            "mate": mate,
            "depth": completed_depth,
            "nodes": self.nodes,
            "time": elapsed,
            "nps": int(self.nodes / elapsed),
            "pv": [pos.uci(m) for m in pv],
        }

    def _root(self, pos, depth, alpha, beta):
        tt_move = None
        entry = self.tt.get(pos.key)
        if entry:
            tt_move = entry[3]
        moves = pos.pseudo_moves(False)
        ordered = self._score_moves(pos, moves, tt_move, 0)
        best_move = None
        best = -INF
        flag = UPPER
        line = []
        legal_count = 0
        for i, m in enumerate(ordered):
            undo = pos.make_move(m)
            mover = pos.turn ^ COLOR_MASK
            ksq = pos.kings[mover]
            if ksq >= 0 and pos.is_attacked(ksq, pos.turn):
                pos.unmake_move(m, undo)
                continue
            legal_count += 1
            if legal_count == 1:
                score = -self._negamax(pos, depth - 1, -beta, -alpha, 1, True)
            else:
                score = -self._negamax(pos, depth - 1, -alpha - 1, -alpha, 1, True)
                if alpha < score < beta:
                    score = -self._negamax(pos, depth - 1, -beta, -alpha, 1, True)
            pos.unmake_move(m, undo)
            if self.stopped:
                break
            if score > best:
                best = score
                best_move = m
                line = [m] + self._pv[1]
                if score > alpha:
                    alpha = score
                    flag = EXACT
                    if alpha >= beta:
                        flag = LOWER
                        break
        if best_move is None:
            best_move = ordered[0]
            best = 0
        self.tt[pos.key] = (depth, flag, best, best_move)
        return best, best_move, line

    # ------------------------------------------------------------ основной поиск
    def _negamax(self, pos: Position, depth: int, alpha: int, beta: int,
                 ply: int, can_null: bool) -> int:
        if self._time_up():
            return 0
        self.nodes += 1
        if ply >= self.MAX_PLY - 2:
            return evaluate(pos)

        in_check = pos.in_check()
        if in_check:
            depth += 1
        if depth <= 0:
            return self._quiesce(pos, alpha, beta, ply)

        if pos.half >= 100:
            return 0

        key = pos.key
        entry = self.tt.get(key)
        tt_move = None
        if entry:
            d, flag, score, tt_move = entry
            if d >= depth:
                if flag == EXACT:
                    return score
                if flag == LOWER:
                    if score >= beta:
                        return score
                elif flag == UPPER:
                    if score <= alpha:
                        return score

        static = evaluate(pos)
        # reverse futility
        if not in_check and depth <= 3 and static - 120 * depth >= beta and can_null:
            return static

        # нулевой ход
        if (can_null and not in_check and depth >= 3 and
                static >= beta and
                self._non_pawn_material(pos, pos.turn) > 0):
            r = 2 + depth // 6
            undo = pos.make_null()
            score = -self._negamax(pos, depth - 1 - r, -beta, -beta + 1,
                                   ply + 1, False)
            pos.unmake_null(undo)
            if self.stopped:
                return 0
            if score >= beta:
                return beta

        moves = pos.pseudo_moves(False)
        ordered = self._score_moves(pos, moves, tt_move, ply)
        best = -INF
        best_move = None
        flag = UPPER
        legal_count = 0
        self._pv[ply + 1] = []

        for i, m in enumerate(ordered):
            is_quiet = pos.board[m[1]] == 0 and m[2] == 0 and m[3] == 0
            undo = pos.make_move(m)
            mover = pos.turn ^ COLOR_MASK
            ksq = pos.kings[mover]
            if ksq >= 0 and pos.is_attacked(ksq, pos.turn):
                pos.unmake_move(m, undo)
                continue
            legal_count += 1

            if legal_count == 1:
                score = -self._negamax(pos, depth - 1, -beta, -alpha, ply + 1, True)
            else:
                reduce = 0
                if depth >= 3 and is_quiet and i >= 3 and not in_check:
                    reduce = 1 if i < 6 else 2
                    if reduce >= depth:
                        reduce = depth - 1
                score = -self._negamax(pos, depth - 1 - reduce, -alpha - 1,
                                       -alpha, ply + 1, True)
                if score > alpha and reduce:
                    score = -self._negamax(pos, depth - 1, -alpha - 1,
                                           -alpha, ply + 1, True)
                if alpha < score < beta:
                    score = -self._negamax(pos, depth - 1, -beta, -alpha,
                                           ply + 1, True)
            pos.unmake_move(m, undo)

            if self.stopped:
                return 0 if best_move is None else best

            if score > best:
                best = score
                best_move = m
                self._pv[ply] = [m] + self._pv[ply + 1]
                if score > alpha:
                    alpha = score
                    flag = EXACT
                    if alpha >= beta:
                        flag = LOWER
                        # фиксируем убийцу и историю для тихих ходов
                        if m[3] == 0 and pos.board[m[1]] == 0:
                            k = self.killers[ply]
                            if k[0] != m:
                                k[1] = k[0]
                                k[0] = m
                            self.history[(m[0], m[1])] = \
                                self.history.get((m[0], m[1]), 0) + depth * depth
                        break

        if legal_count == 0:
            return -MATE + ply if in_check else 0

        self.tt[key] = (depth, flag, best, best_move)
        return best

    # ------------------------------------------------------------- quiescence
    def _quiesce(self, pos: Position, alpha: int, beta: int, ply: int) -> int:
        if self._time_up():
            return 0
        self.nodes += 1
        if ply >= self.MAX_PLY - 2:
            return evaluate(pos)

        stand = evaluate(pos)
        if stand >= beta:
            return beta
        if stand > alpha:
            alpha = stand

        moves = pos.pseudo_moves(True)
        scored = []
        board = pos.board
        for m in moves:
            frm, to, promo, flag = m
            victim = board[to]
            if victim:
                vt = victim & 7
                s = 100 * MATERIAL[vt] - MATERIAL[board[frm] & 7]
            elif flag == FL_EP:
                s = 100 * MATERIAL[PAWN] - MATERIAL[PAWN]
            else:
                s = 0
            if promo == QUEEN:
                s += 800
            scored.append((s, m))
        scored.sort(key=lambda x: -x[0])

        for _, m in scored:
            frm, to, promo, flag = m
            victim = board[to]
            # delta pruning
            gain = MATERIAL[victim & 7] if victim else 0
            if promo == QUEEN:
                gain += MATERIAL[QUEEN] - MATERIAL[PAWN]
            if stand + gain + 200 < alpha:
                continue
            undo = pos.make_move(m)
            mover = pos.turn ^ COLOR_MASK
            ksq = pos.kings[mover]
            if ksq >= 0 and pos.is_attacked(ksq, pos.turn):
                pos.unmake_move(m, undo)
                continue
            score = -self._quiesce(pos, -beta, -alpha, ply + 1)
            pos.unmake_move(m, undo)
            if self.stopped:
                return stand
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha


# ============================================================== дебютная книга
BOOK_LINES = [
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6 c2c3 e8g8",
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 c1e3 e7e5 d4b3 c8e6 f2f3 f8e7",
    "e2e4 e7e6 d2d4 d7d5 b1c3 f8e7 g1f3 g8f6 e4e5 f6d7 f1d3 c7c5 d4c5 d7c5 d3e4",
    "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4 c8f5 e4g3 f5g6 h2h4 h7h5 g1f3 b8d7 h4h5 g6h7",
    "e2e4 d7d5 e4d5 d8d5 b1c3 d5a5 d2d4 g8f6 g1f3 c8f5 c1f4 e7e6 f1d3 f8b4 e1g1",
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 g1f3 f8e7 c1f5 e8g8 e2e3 c7c6 f1d3 b8d7 e1g1",
    "d2d4 g8f6 c2c4 e7e6 g1f3 d7d5 b1c3 f8e7 c1f5 e8g8 e2e3 c7c6 f1d3 b8d7 e1g1",
    "d2d4 d7d5 g1f3 g8f6 c2c4 e7e6 g2g3 f8e7 f1g2 e8g8 e1g1 d5c4 d1a4 a7a6 a4c4",
    "c2c4 e7e5 b1c3 g8f6 g1f3 b8c6 g2g3 f8b4 f1g2 e8g8 e1g1 b4c3 b2c3 d7d6 d2d4",
    "c2c4 g8f6 b1c3 e7e6 g1f3 d7d5 d2d4 f8e7 c1f5 e8g8 e2e3 c7c5 a2a3 b8c6 f1d3",
    "g1f3 d7d5 d2d4 g8f6 c2c4 e7e6 b1c3 f8e7 c1f5 e8g8 e2e3 c7c6 f1d3 b8d7 e1g1",
    "g1f3 g8f6 c2c4 g7g6 b1c3 f8g7 d2d4 d7d5 c4d5 f6d5 e2e4 d5c3 b2c3 e8g8 f1e2",
    "e2e4 e7e5 f1c4 g8f6 d2d4 e5d4 g1f3 f8c5 e1g1 d7d6 f3g5 d6d5 e4d5",
]
BOOK_PLY_LIMIT = 16


def _build_book():
    book = {}
    for line in BOOK_LINES:
        pos = Position.start()
        for uci in line.split():
            m = pos.move_from_uci(uci)
            if m is None:
                break
            key = (pos.placement(), pos.turn)
            book.setdefault(key, [])
            if m not in book[key]:
                book[key].append(m)
            pos.make_move(m)
    return book


BOOK = _build_book()


# ========================================================== обёртка Stockfish
def find_stockfish() -> str | None:
    import glob
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(here, "stockfish.exe"),
        os.path.join(here, "stockfish", "stockfish.exe"),
        os.path.join(here, "bin", "stockfish.exe"),
        os.path.join(os.getcwd(), "stockfish.exe"),
        shutil.which("stockfish") or "",
        shutil.which("stockfish.exe") or "",
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    for folder in [here, os.path.join(here, "stockfish"), os.path.join(here, "bin"), os.getcwd(), os.path.dirname(here)]:
        if os.path.isdir(folder):
            for p in glob.glob(os.path.join(folder, "*stockfish*.exe")):
                if os.path.isfile(p):
                    return p
            for p in glob.glob(os.path.join(folder, "**", "*stockfish*.exe"), recursive=True):
                if os.path.isfile(p):
                    return p
    return None


class StockfishEngine:
    """Надёжный UCI-клиент с авто-восстановлением, MultiPV, контролем силы (Elo) и защитой от [Errno 22]."""

    def __init__(self, path: str):
        self.path = path
        self.proc = None
        self.lock = threading.RLock()
        self.name = "stockfish"
        self.multipv = 1
        self.elo = 0
        self._start_process()

    def _start_process(self):
        with self.lock:
            if self.proc is not None:
                try:
                    self.proc.kill()
                    self.proc.wait(timeout=1)
                except Exception:
                    pass
                self.proc = None

            self.proc = subprocess.Popen(
                [self.path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            )
            self._send_raw("uci")
            self._wait_for("uciok", timeout=5.0)
            if getattr(self, "multipv", 1) > 1:
                self._send_raw(f"setoption name MultiPV value {self.multipv}")
            if getattr(self, "elo", 0) > 0:
                self._apply_elo(self.elo)
            self._send_raw("isready")
            self._wait_for("readyok", timeout=5.0)

    def set_multipv(self, count: int = 1):
        with self.lock:
            self.multipv = max(1, min(5, int(count)))
            self._send(f"setoption name MultiPV value {self.multipv}")

    def set_elo(self, elo: int | None = None):
        with self.lock:
            self.elo = int(elo) if elo else 0
            self._apply_elo(self.elo)

    def _apply_elo(self, elo: int):
        if elo <= 0 or elo >= 3000:
            self._send("setoption name UCI_LimitStrength value false")
            self._send("setoption name Skill Level value 20")
        else:
            target_elo = max(1320, min(3190, int(elo)))
            self._send("setoption name UCI_LimitStrength value true")
            self._send(f"setoption name UCI_Elo value {target_elo}")
            skill = max(0, min(20, int((target_elo - 1200) / 90)))
            self._send(f"setoption name Skill Level value {skill}")

    def _send_raw(self, line: str):
        if self.proc is None or self.proc.stdin is None:
            raise OSError("Stockfish process not running")
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _send(self, line: str):
        try:
            if self.proc is None or self.proc.poll() is not None:
                self._start_process()
            self._send_raw(line)
        except (OSError, BrokenPipeError):
            self._start_process()
            self._send_raw(line)

    def _wait_for(self, prefix: str, timeout: float = 30.0) -> str:
        end = time.time() + timeout
        last = ""
        while time.time() < end:
            try:
                line = self.proc.stdout.readline()
            except Exception:
                break
            if not line:
                break
            line = line.strip()
            last = line
            if line.startswith(prefix):
                return line
        return last

    def analyze(self, pos: Position, movetime_ms: int = 1500,
                stop_event: threading.Event | None = None) -> dict:
        with self.lock:
            t0 = time.time()
            if self.proc is None or self.proc.poll() is not None:
                self._start_process()

            # Дренируем и синхронизируем процесс перед новым анализом
            try:
                self._send("stop")
                self._send("isready")
                self._wait_for("readyok", timeout=0.8)
            except Exception:
                self._start_process()

            fen = pos.to_fen()
            try:
                self._send(f"position fen {fen}")
                self._send(f"go movetime {int(movetime_ms)}")
            except Exception:
                self._start_process()
                self._send(f"position fen {fen}")
                self._send(f"go movetime {int(movetime_ms)}")

            best = None
            score_cp = 0
            depth = 0
            pv = []
            multipv_data = {}
            end = time.time() + movetime_ms / 1000.0 + 3.0

            while time.time() < end:
                if stop_event is not None and stop_event.is_set():
                    try:
                        self._send("stop")
                    except Exception:
                        pass
                    end = min(end, time.time() + 0.3)

                try:
                    line = self.proc.stdout.readline()
                except Exception:
                    break

                if not line:
                    break
                line = line.strip()

                if "CRITICAL ERROR" in line:
                    break

                if line.startswith("info ") and " score " in line:
                    parts = line.split()
                    try:
                        mpv = 1
                        if "multipv" in parts:
                            mpv = int(parts[parts.index("multipv") + 1])

                        i = parts.index("score")
                        if parts[i + 1] == "cp":
                            sc = int(parts[i + 2])
                        elif parts[i + 1] == "mate":
                            sc = (1 if int(parts[i + 2]) > 0 else -1) * (MATE - 1)
                        else:
                            sc = 0

                        d = int(parts[parts.index("depth") + 1]) if "depth" in parts else 0
                        p_val = parts[parts.index("pv") + 1:] if "pv" in parts else []

                        multipv_data[mpv] = {
                            "score": sc,
                            "depth": d,
                            "pv": p_val,
                            "uci": p_val[0] if p_val else None
                        }
                        if mpv == 1:
                            score_cp = sc
                            depth = d
                            pv = p_val
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("bestmove"):
                    parts = line.split()
                    best = parts[1] if len(parts) > 1 else None
                    break

            lines = []
            for mpv in sorted(multipv_data.keys()):
                info = multipv_data[mpv]
                mv_uci = info.get("uci")
                if mv_uci and mv_uci != "(none)":
                    mv_obj = pos.move_from_uci(mv_uci)
                    if mv_obj is not None:
                        lines.append({
                            "rank": mpv,
                            "move": mv_obj,
                            "uci": mv_uci,
                            "san": pos.san(mv_obj),
                            "score": info.get("score", 0),
                            "depth": info.get("depth", 0),
                            "pv": info.get("pv", [])
                        })

            move = None
            if lines:
                move = lines[0]["move"]
                best = lines[0]["uci"]
                score_cp = lines[0]["score"]
            elif best and best != "(none)":
                move = pos.move_from_uci(best)
                if move:
                    lines.append({
                        "rank": 1,
                        "move": move,
                        "uci": best,
                        "san": pos.san(move),
                        "score": score_cp,
                        "depth": depth,
                        "pv": pv
                    })

            return {"move": move, "uci": best, "san": pos.san(move) if move else None,
                    "score": score_cp, "mate": 0, "depth": depth,
                    "nodes": 0, "time": time.time() - t0, "nps": 0, "pv": pv,
                    "lines": lines}

    def close(self):
        with self.lock:
            try:
                if self.proc is not None:
                    self._send_raw("quit")
                    self.proc.wait(timeout=2)
            except Exception:
                try:
                    if self.proc is not None:
                        self.proc.kill()
                except Exception:
                    pass
            self.proc = None


class Engine:
    """Единая точка входа: stockfish, если он есть, иначе встроенный поиск."""

    def __init__(self, prefer_builtin: bool = False, use_book: bool = True):
        self.native = Searcher("builtin")
        self.sf = None
        self.use_book = use_book
        if not prefer_builtin:
            path = find_stockfish()
            if path:
                try:
                    self.sf = StockfishEngine(path)
                except Exception:
                    self.sf = None
        self.name = self.sf.name if self.sf else "builtin"

    # ----------------------------------------------------------- дебютная книга
    @staticmethod
    def book_move(pos: Position):
        key = (pos.placement(), pos.turn)
        moves = BOOK.get(key)
        if not moves:
            return None
        ply = max(0, (pos.full - 1) * 2 + (0 if pos.turn == WHITE else 1))
        if ply > BOOK_PLY_LIMIT:
            return None
        import random
        return random.choice(moves)

    def analyze(self, pos: Position, movetime_ms: int = 1500,
                stop_event: threading.Event | None = None) -> dict:
        if self.sf is None and self.use_book:
            mv = self.book_move(pos)
            if mv is not None:
                return {"move": mv, "uci": pos.uci(mv), "san": pos.san(mv),
                        "score": 0, "mate": 0, "depth": 0, "nodes": 0,
                        "time": 0.0, "nps": 0, "pv": [pos.uci(mv)],
                        "book": True}
        if self.sf is not None:
            return self.sf.analyze(pos, movetime_ms, stop_event)
        self.native.external_stop = stop_event
        return self.native.search(pos, movetime_ms=movetime_ms)

    def set_multipv(self, count: int = 1):
        if self.sf is not None:
            self.sf.set_multipv(count)

    def set_elo(self, elo: int | None = None):
        if self.sf is not None:
            self.sf.set_elo(elo)

    def close(self):
        if self.sf is not None:
            self.sf.close()
