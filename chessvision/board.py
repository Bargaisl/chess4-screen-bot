"""Шахматная доска: 0x88-представление, генерация ходов, FEN, SAN, Zobrist,
и строгая валидация позиций для защиты от сбоев распознавания.
"""
from __future__ import annotations

WHITE = 8
BLACK = 16
COLOR_MASK = 24

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6
NO_PIECE = 0

FL_NORMAL = 0
FL_DOUBLE = 1
FL_EP = 2
FL_KCASTLE = 3
FL_QCASTLE = 4

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

PIECE_CHARS = {
    WHITE | PAWN: "P", WHITE | KNIGHT: "N", WHITE | BISHOP: "B",
    WHITE | ROOK: "R", WHITE | QUEEN: "Q", WHITE | KING: "K",
    BLACK | PAWN: "p", BLACK | KNIGHT: "n", BLACK | BISHOP: "b",
    BLACK | ROOK: "r", BLACK | QUEEN: "q", BLACK | KING: "k",
}
CHAR_TO_PIECE = {v: k for k, v in PIECE_CHARS.items()}

KNIGHT_D = (33, 31, 18, 14, -14, -18, -31, -33)
BISHOP_D = (17, 15, -15, -17)
ROOK_D = (16, 1, -1, -16)
KING_D = (17, 16, 15, 1, -1, -15, -16, -17)

_FILES = "abcdefgh"
PROMO_UCI = {KNIGHT: "n", BISHOP: "b", ROOK: "r", QUEEN: "q"}
PROMO_SAN = {KNIGHT: "N", BISHOP: "B", ROOK: "R", QUEEN: "Q"}
PIECE_LETTER = {KNIGHT: "N", BISHOP: "B", ROOK: "R", QUEEN: "Q", KING: "K"}


def sq_name(sq: int) -> str:
    return _FILES[sq & 7] + str((sq >> 4) + 1)


def name_sq(name: str) -> int:
    return (int(name[1]) - 1) * 16 + _FILES.index(name[0].lower())


def _rays(deltas, max_len):
    out = []
    for sq in range(128):
        if sq & 0x88:
            out.append(tuple(() for _ in deltas))
            continue
        row = []
        for d in deltas:
            path = []
            t = sq
            for _ in range(max_len):
                t += d
                if t & 0x88:
                    break
                path.append(t)
            row.append(tuple(path))
        out.append(tuple(row))
    return tuple(out)


RAY_DIAG = _rays(BISHOP_D, 7)
RAY_ORTH = _rays(ROOK_D, 7)
RAY_KNIGHT = tuple(tuple(sq + d for d in KNIGHT_D if not ((sq + d) & 0x88))
                   for sq in range(128))
RAY_KING = tuple(tuple(sq + d for d in KING_D if not ((sq + d) & 0x88))
                 for sq in range(128))

WHITE_PAWN_ATTACKS = (-15, -17)
BLACK_PAWN_ATTACKS = (15, 17)


# ---------------------------------------------------------------- Zobrist
def _build_zobrist():
    import random
    rnd = random.Random(0x5E3C5EED)
    piece = [[rnd.getrandbits(64) for _ in range(128)] for _ in range(32)]
    ep = [rnd.getrandbits(64) for _ in range(8)]
    castle = [rnd.getrandbits(64) for _ in range(16)]
    side = rnd.getrandbits(64)
    return piece, ep, castle, side


Z_PIECE, Z_EP, Z_CASTLE, Z_SIDE = _build_zobrist()
MASK64 = (1 << 64) - 1


class Position:
    __slots__ = ("board", "turn", "castling", "ep", "half", "full", "kings", "key")

    def __init__(self):
        self.board = [0] * 128
        self.turn = WHITE
        self.castling = 0            # 1=K, 2=Q, 4=k, 8=q
        self.ep = -1
        self.half = 0
        self.full = 1
        self.kings = {WHITE: -1, BLACK: -1}
        self.key = 0

    @classmethod
    def from_fen(cls, fen: str) -> "Position":
        pos = cls()
        parts = fen.strip().split()
        if not parts:
            raise ValueError("пустой FEN")
        rows = parts[0].split("/")
        if len(rows) != 8:
            raise ValueError("некорректный FEN: ожидалось 8 рядов")
        for r, row in enumerate(rows):
            rank = 7 - r
            file = 0
            for ch in row:
                if ch.isdigit():
                    file += int(ch)
                    continue
                proto = CHAR_TO_PIECE.get(ch.upper())
                if proto is None or file > 7:
                    raise ValueError(f"некорректный символ FEN: {ch!r}")
                color = WHITE if ch.isupper() else BLACK
                p = color | (proto & 7)
                sq = rank * 16 + file
                pos.board[sq] = p
                if (p & 7) == KING:
                    pos.kings[color] = sq
                file += 1
        if len(parts) > 1:
            pos.turn = WHITE if parts[1].lower().startswith("w") else BLACK
        if len(parts) > 2 and parts[2] != "-":
            if "K" in parts[2]: pos.castling |= 1
            if "Q" in parts[2]: pos.castling |= 2
            if "k" in parts[2]: pos.castling |= 4
            if "q" in parts[2]: pos.castling |= 8
        if len(parts) > 3 and parts[3] not in ("-", ""):
            pos.ep = name_sq(parts[3])
        if len(parts) > 4:
            try: pos.half = int(parts[4])
            except ValueError: pass
        if len(parts) > 5:
            try: pos.full = int(parts[5])
            except ValueError: pass
        pos.compute_key()
        return pos

    @classmethod
    def start(cls) -> "Position":
        return cls.from_fen(START_FEN)

    def clone(self) -> "Position":
        p = Position()
        p.board = list(self.board)
        p.turn = self.turn
        p.castling = self.castling
        p.ep = self.ep
        p.half = self.half
        p.full = self.full
        p.kings = dict(self.kings)
        p.key = self.key
        return p

    def placement(self) -> str:
        rows = []
        for rank in range(7, -1, -1):
            empty = 0
            row = ""
            for file in range(8):
                sq = rank * 16 + file
                p = self.board[sq]
                if p == 0:
                    empty += 1
                else:
                    if empty:
                        row += str(empty)
                        empty = 0
                    row += PIECE_CHARS[p]
            if empty:
                row += str(empty)
            rows.append(row)
        return "/".join(rows)

    def to_fen(self) -> str:
        c = ""
        if self.castling & 1: c += "K"
        if self.castling & 2: c += "Q"
        if self.castling & 4: c += "k"
        if self.castling & 8: c += "q"
        if not c: c = "-"
        ep = sq_name(self.ep) if self.ep >= 0 else "-"
        turn_ch = "w" if self.turn == WHITE else "b"
        return f"{self.placement()} {turn_ch} {c} {ep} {self.half} {self.full}"

    def compute_key(self) -> int:
        k = 0
        for sq in range(128):
            if sq & 0x88:
                continue
            p = self.board[sq]
            if p:
                k ^= Z_PIECE[p][sq]
        if self.turn == BLACK:
            k ^= Z_SIDE
        k ^= Z_CASTLE[self.castling]
        if self.ep >= 0:
            k ^= Z_EP[self.ep & 7]
        self.key = k & MASK64
        return self.key

    def make_null(self):
        undo = (self.ep, self.half, self.key, self.turn)
        if self.ep >= 0:
            self.key ^= Z_EP[self.ep & 7]
        self.ep = -1
        self.turn = BLACK if self.turn == WHITE else WHITE
        self.key = (self.key ^ Z_SIDE) & MASK64
        return undo

    def unmake_null(self, undo):
        self.ep, self.half, self.key, self.turn = undo

    def is_attacked(self, sq: int, by_color: int) -> bool:
        b = self.board
        if by_color == WHITE:
            for d in WHITE_PAWN_ATTACKS:
                from_sq = sq + d
                if not (from_sq & 0x88) and b[from_sq] == (WHITE | PAWN):
                    return True
        else:
            for d in BLACK_PAWN_ATTACKS:
                from_sq = sq + d
                if not (from_sq & 0x88) and b[from_sq] == (BLACK | PAWN):
                    return True

        for from_sq in RAY_KNIGHT[sq]:
            if b[from_sq] == (by_color | KNIGHT):
                return True

        for to in RAY_KING[sq]:
            if b[to] == (by_color | KING):
                return True

        for i in range(4):
            for from_sq in RAY_DIAG[sq][i]:
                p = b[from_sq]
                if p == 0:
                    continue
                if p in (by_color | BISHOP, by_color | QUEEN):
                    return True
                break

        for i in range(4):
            for from_sq in RAY_ORTH[sq][i]:
                p = b[from_sq]
                if p == 0:
                    continue
                if p in (by_color | ROOK, by_color | QUEEN):
                    return True
                break

        return False

    def in_check(self, color: int | None = None) -> bool:
        c = self.turn if color is None else color
        ksq = self.kings[c]
        if ksq < 0:
            return False
        them = BLACK if c == WHITE else WHITE
        return self.is_attacked(ksq, them)

    def _add_pawn_move(self, frm: int, to: int, promo: bool, flag: int, add):
        if promo:
            for p in (QUEEN, KNIGHT, ROOK, BISHOP):
                add((frm, to, p, flag))
        else:
            add((frm, to, 0, flag))

    def pseudo_moves(self, captures_only: bool = False):
        b = self.board
        us = self.turn
        them = BLACK if us == WHITE else WHITE
        moves = []
        add = moves.append

        for sq in range(128):
            if sq & 0x88:
                continue
            p = b[sq]
            if p == 0 or (p & COLOR_MASK) != us:
                continue
            t = p & 7
            if t == PAWN:
                d = 16 if us == WHITE else -16
                from_rank = sq >> 4
                start_rank = 1 if us == WHITE else 6
                promo_rank = 6 if us == WHITE else 1
                if not captures_only:
                    fwd = sq + d
                    if not fwd & 0x88 and b[fwd] == 0:
                        self._add_pawn_move(sq, fwd, from_rank == promo_rank,
                                            FL_NORMAL, add)
                        if from_rank == start_rank and b[fwd + d] == 0:
                            add((sq, fwd + d, 0, FL_DOUBLE))
                for fd in (-1, 1):
                    to = sq + d + fd
                    if to & 0x88:
                        continue
                    q = b[to]
                    if q and (q & COLOR_MASK) == them:
                        self._add_pawn_move(sq, to, from_rank == promo_rank,
                                            FL_NORMAL, add)
                    elif self.ep >= 0 and to == self.ep:
                        add((sq, to, 0, FL_EP))
                if captures_only and from_rank == promo_rank:
                    fwd = sq + d
                    if not fwd & 0x88 and b[fwd] == 0:
                        for pr in (QUEEN, ROOK, BISHOP, KNIGHT):
                            add((sq, fwd, pr, FL_NORMAL))
            elif t in (KNIGHT, KING):
                for to in (RAY_KNIGHT[sq] if t == KNIGHT else RAY_KING[sq]):
                    q = b[to]
                    if q == 0:
                        if not captures_only:
                            add((sq, to, 0, FL_NORMAL))
                    elif (q & COLOR_MASK) == them:
                        add((sq, to, 0, FL_NORMAL))
            else:
                diag = t in (BISHOP, QUEEN)
                orth = t in (ROOK, QUEEN)
                if diag:
                    for i in range(4):
                        for to in RAY_DIAG[sq][i]:
                            q = b[to]
                            if q == 0:
                                if not captures_only:
                                    add((sq, to, 0, FL_NORMAL))
                            else:
                                if (q & COLOR_MASK) == them:
                                    add((sq, to, 0, FL_NORMAL))
                                break
                if orth:
                    for i in range(4):
                        for to in RAY_ORTH[sq][i]:
                            q = b[to]
                            if q == 0:
                                if not captures_only:
                                    add((sq, to, 0, FL_NORMAL))
                            else:
                                if (q & COLOR_MASK) == them:
                                    add((sq, to, 0, FL_NORMAL))
                                break

        if not captures_only:
            self._castling_moves(us, add)
        return moves

    def _castling_moves(self, us, add):
        if self.in_check(us):
            return
        b = self.board
        them = BLACK if us == WHITE else WHITE
        if us == WHITE:
            if (self.castling & 1) and b[4] == WHITE | KING and b[7] == WHITE | ROOK:
                if b[5] == 0 and b[6] == 0 and not self.is_attacked(5, them) and not self.is_attacked(6, them):
                    add((4, 6, 0, FL_KCASTLE))
            if (self.castling & 2) and b[4] == WHITE | KING and b[0] == WHITE | ROOK:
                if b[1] == 0 and b[2] == 0 and b[3] == 0 and not self.is_attacked(3, them) and not self.is_attacked(2, them):
                    add((4, 2, 0, FL_QCASTLE))
        else:
            if (self.castling & 4) and b[116] == BLACK | KING and b[119] == BLACK | ROOK:
                if b[117] == 0 and b[118] == 0 and not self.is_attacked(117, them) and not self.is_attacked(118, them):
                    add((116, 118, 0, FL_KCASTLE))
            if (self.castling & 8) and b[116] == BLACK | KING and b[112] == BLACK | ROOK:
                if b[113] == 0 and b[114] == 0 and b[115] == 0 and not self.is_attacked(115, them) and not self.is_attacked(114, them):
                    add((116, 114, 0, FL_QCASTLE))

    def legal_moves(self):
        out = []
        for m in self.pseudo_moves(False):
            undo = self.make_move(m)
            mover = self.turn ^ COLOR_MASK
            ksq = self.kings[mover]
            if ksq >= 0 and not self.is_attacked(ksq, self.turn):
                out.append(m)
            self.unmake_move(m, undo)
        return out

    def legal_captures(self):
        out = []
        for m in self.pseudo_moves(True):
            undo = self.make_move(m)
            mover = self.turn ^ COLOR_MASK
            ksq = self.kings[mover]
            if ksq >= 0 and not self.is_attacked(ksq, self.turn):
                out.append(m)
            self.unmake_move(m, undo)
        return out

    def is_legal(self, m) -> bool:
        undo = self.make_move(m)
        mover = self.turn ^ COLOR_MASK
        ksq = self.kings[mover]
        ok = ksq >= 0 and not self.is_attacked(ksq, self.turn)
        self.unmake_move(m, undo)
        return ok

    def make_move(self, m):
        frm, to, promo, flag = m
        b = self.board
        p = b[frm]
        cap = b[to]
        us = self.turn
        them = BLACK if us == WHITE else WHITE
        undo = (cap, self.castling, self.ep, self.half, self.key)

        b[frm] = 0
        b[to] = (us | promo) if promo else p
        if (p & 7) == KING:
            self.kings[us] = to

        if flag == FL_DOUBLE:
            self.ep = (frm + to) // 2
        elif flag == FL_EP:
            cap_sq = to - (16 if us == WHITE else -16)
            cap = b[cap_sq]
            b[cap_sq] = 0
            self.ep = -1
        elif flag == FL_KCASTLE:
            rfrm, rto = (7, 5) if us == WHITE else (119, 117)
            b[rto] = b[rfrm]
            b[rfrm] = 0
            self.ep = -1
        elif flag == FL_QCASTLE:
            rfrm, rto = (0, 3) if us == WHITE else (112, 115)
            b[rto] = b[rfrm]
            b[rfrm] = 0
            self.ep = -1
        else:
            self.ep = -1

        if p == WHITE | KING: self.castling &= ~3
        elif p == BLACK | KING: self.castling &= ~12
        if frm == 0 or to == 0: self.castling &= ~2
        if frm == 7 or to == 7: self.castling &= ~1
        if frm == 112 or to == 112: self.castling &= ~8
        if frm == 119 or to == 119: self.castling &= ~4

        if (p & 7) == PAWN or cap:
            self.half = 0
        else:
            self.half += 1

        if us == BLACK:
            self.full += 1
        self.turn = them
        self.compute_key()
        return undo

    def unmake_move(self, m, undo):
        frm, to, promo, flag = m
        cap, self.castling, self.ep, self.half, self.key = undo
        them = self.turn
        us = BLACK if them == WHITE else WHITE
        self.turn = us
        if us == BLACK:
            self.full -= 1

        b = self.board
        p = (us | PAWN) if promo else b[to]
        b[frm] = p
        b[to] = cap
        if (p & 7) == KING:
            self.kings[us] = frm

        if flag == FL_EP:
            cap_sq = to - (16 if us == WHITE else -16)
            b[cap_sq] = them | PAWN
            b[to] = 0
        elif flag == FL_KCASTLE:
            rfrm, rto = (7, 5) if us == WHITE else (119, 117)
            b[rfrm] = b[rto]
            b[rto] = 0
        elif flag == FL_QCASTLE:
            rfrm, rto = (0, 3) if us == WHITE else (112, 115)
            b[rfrm] = b[rto]
            b[rto] = 0

    def uci(self, m) -> str:
        frm, to, promo, _ = m
        s = sq_name(frm) + sq_name(to)
        if promo:
            s += PROMO_UCI[promo]
        return s

    def san(self, m) -> str:
        frm, to, promo, flag = m
        if flag == FL_KCASTLE:
            return "O-O"
        if flag == FL_QCASTLE:
            return "O-O-O"
        p = self.board[frm]
        t = p & 7
        cap = bool(self.board[to]) or flag == FL_EP
        if t == PAWN:
            s = (_FILES[frm & 7] + "x" if cap else "") + sq_name(to)
            if promo:
                s += "=" + PROMO_SAN[promo]
        else:
            letter = PIECE_LETTER[t]
            others = [m2[0] for m2 in self.pseudo_moves(False)
                      if m2[1] == to and m2[0] != frm and self.board[m2[0]] == p]
            disamb = ""
            if others:
                if not any((o & 7) == (frm & 7) for o in others):
                    disamb = _FILES[frm & 7]
                elif not any((o >> 4) == (frm >> 4) for o in others):
                    disamb = str((frm >> 4) + 1)
                else:
                    disamb = sq_name(frm)
            s = letter + disamb + ("x" if cap else "") + sq_name(to)
        undo = self.make_move(m)
        if self.in_check():
            s += "#" if not self.legal_moves() else "+"
        self.unmake_move(m, undo)
        return s

    def move_from_uci(self, text: str):
        text = text.strip().lower()
        for m in self.legal_moves():
            if self.uci(m) == text:
                return m
        return None

    def game_status(self, rep_keys=()) -> str:
        """'checkmate' | 'stalemate' | 'draw50' | 'repetition' | 'material' | 'play'"""
        moves = self.legal_moves()
        if not moves:
            return "checkmate" if self.in_check() else "stalemate"
        if self.half >= 100:
            return "draw50"
        if rep_keys.count(self.key) >= 2:
            return "repetition"
        counts = {}
        for sq in range(128):
            if sq & 0x88:
                continue
            p = self.board[sq]
            if p:
                counts[p & 7] = counts.get(p & 7, 0) + 1
        if not counts.get(PAWN) and not counts.get(ROOK) and not counts.get(QUEEN):
            minors = counts.get(KNIGHT, 0) + counts.get(BISHOP, 0)
            if minors <= 1:
                return "material"
        return "play"



# --------------------------------------------------- проверка расстановки
def expand_placement(placement: str) -> list[list[str]] | None:
    rows = placement.split("/")
    if len(rows) != 8:
        return None
    out = []
    for r in rows:
        line = []
        for ch in r:
            if ch.isdigit():
                line.extend(["1"] * int(ch))
            elif ch in CHAR_TO_PIECE:
                line.append(ch)
            else:
                return None
        if len(line) != 8:
            return None
        out.append(line)
    return out


def placement_issues(placement: str | None) -> list[str]:
    """Строгая проверка правдоподобия шахматной позиции.

    Защищает от любых сбоев детектора (сдвиг сетки, шум, мусор):
    - Ровно 1 белый и 1 чёрный король
    - Короли не могут стоять на соседних клетках
    - Пешки не могут стоять на 1-й и 8-й горизонталях
    - Максимум 8 пешек у каждой стороны
    - Максимум 16 фигур у каждой стороны
    - Ограничение превращений: сумма лишних тяжёлых/лёгких фигур
      не может превышать число исчезнувших пешек (8 - пешки)!
    """
    if not placement:
        return ["позиция не распознана"]
    grid = expand_placement(placement)
    if grid is None:
        return ["расстановка не 8x8 или содержит не-фигуры"]
    flat = [ch for row in grid for ch in row]
    issues = []

    # 1. Короли
    white_kings = []
    black_kings = []
    for r in range(8):
        for c in range(8):
            ch = grid[r][c]
            if ch == "K":
                white_kings.append((r, c))
            elif ch == "k":
                black_kings.append((r, c))

    if len(white_kings) != 1:
        issues.append(f"королей белых: {len(white_kings)} (нужен ровно 1)")
    if len(black_kings) != 1:
        issues.append(f"королей чёрных: {len(black_kings)} (нужен ровно 1)")

    if len(white_kings) == 1 and len(black_kings) == 1:
        wr, wc = white_kings[0]
        br, bc = black_kings[0]
        if abs(wr - br) <= 1 and abs(wc - bc) <= 1:
            issues.append("короли стоят вплотную друг к другу")

    # 2. Пешки на крайних горизонталях
    for i in (0, 7):
        row = grid[i]
        n = sum(1 for c in row if c in "pP")
        if n:
            issues.append(f"пешка(и) на {8 - i}-й горизонтали")

    # 3. Число пешек и общее число фигур
    w_p = flat.count("P")
    b_p = flat.count("p")
    if w_p > 8:
        issues.append(f"пешек белых: {w_p} (максимум 8)")
    if b_p > 8:
        issues.append(f"пешек чёрных: {b_p} (максимум 8)")

    w_total = sum(flat.count(c) for c in "PNBRQK")
    b_total = sum(flat.count(c) for c in "pnbrqk")
    if w_total > 16:
        issues.append(f"фигур белых: {w_total} (максимум 16)")
    if b_total > 16:
        issues.append(f"фигур чёрных: {b_total} (максимум 16)")

    # 4. Проверка допустимых превращений (Promotion Limits)
    w_promos_max = max(0, 8 - w_p)
    w_knights = flat.count("N")
    w_bishops = flat.count("B")
    w_rooks = flat.count("R")
    w_queens = flat.count("Q")
    w_extra = (max(0, w_knights - 2) + max(0, w_bishops - 2) +
               max(0, w_rooks - 2) + max(0, w_queens - 1))
    if w_extra > w_promos_max:
        issues.append(f"невозможное число белых фигур: {w_bishops} слонов, "
                      f"{w_queens} ферзей при {w_p} пешках")

    b_promos_max = max(0, 8 - b_p)
    b_knights = flat.count("n")
    b_bishops = flat.count("b")
    b_rooks = flat.count("r")
    b_queens = flat.count("q")
    b_extra = (max(0, b_knights - 2) + max(0, b_bishops - 2) +
               max(0, b_rooks - 2) + max(0, b_queens - 1))
    if b_extra > b_promos_max:
        issues.append(f"невозможное число чёрных фигур: {b_bishops} слонов, "
                      f"{b_queens} ферзей при {b_p} пешках")

    # 5. Защита от артефактов компьютерного зрения (скопление однотипных тяжёлых/лёгких фигур)
    for r_idx in range(8):
        row_b_white = sum(1 for c in grid[r_idx] if c == "B")
        if row_b_white >= 4:
            issues.append(f"артефакт зрения: {row_b_white} белых слонов на одной горизонтали")
        row_b_black = sum(1 for c in grid[r_idx] if c == "b")
        if row_b_black >= 4:
            issues.append(f"артефакт зрения: {row_b_black} чёрных слонов на одной горизонтали")

    return issues


def find_reaching_moves(old: Position, new_placement: str):
    out = []
    for m in old.legal_moves():
        undo = old.make_move(m)
        if old.placement() == new_placement:
            out.append(m)
        old.unmake_move(m, undo)
    return out


def infer_move(old: Position, new_placement: str):
    cands = find_reaching_moves(old, new_placement)
    if len(cands) == 1:
        return cands[0]
    if not cands:
        return None
    for m in cands:
        if m[2] in (0, QUEEN):
            return m
    return cands[0]


def side_that_moved(old_placement: str, new_placement: str) -> int | None:
    """Чья сторона сделала изменения между двумя расстановками."""
    a = expand_placement(old_placement)
    b = expand_placement(new_placement)
    if a is None or b is None:
        return None
    arrived = {WHITE: False, BLACK: False}
    for i in range(8):
        for j in range(8):
            ch = b[i][j]
            if ch == "1" or a[i][j] == ch:
                continue
            arrived[WHITE if ch.isupper() else BLACK] = True
    if arrived[WHITE] and not arrived[BLACK]:
        return WHITE
    if arrived[BLACK] and not arrived[WHITE]:
        return BLACK
    return None



def board_from_placement(placement: str, turn: int, castling: int = 0) -> Position:
    pos = Position.from_fen(f"{placement} {'w' if turn == WHITE else 'b'} - - 0 1")
    rights = 0
    if pos.board[4] == WHITE | KING and pos.board[7] == WHITE | ROOK: rights |= 1
    if pos.board[4] == WHITE | KING and pos.board[0] == WHITE | ROOK: rights |= 2
    if pos.board[116] == BLACK | KING and pos.board[119] == BLACK | ROOK: rights |= 4
    if pos.board[116] == BLACK | KING and pos.board[112] == BLACK | ROOK: rights |= 8
    pos.castling = rights if castling == 0 else castling
    pos.compute_key()
    return pos


def perft(pos: Position, depth: int) -> int:
    if depth <= 0:
        return 1
    total = 0
    for m in pos.legal_moves():
        undo = pos.make_move(m)
        total += perft(pos, depth - 1)
        pos.unmake_move(m, undo)
    return total


def perft_bulk(pos: Position, depth: int) -> int:
    if depth <= 0:
        return 1
    if depth == 1:
        return len(pos.legal_moves())
    total = 0
    for m in pos.legal_moves():
        undo = pos.make_move(m)
        total += perft_bulk(pos, depth - 1)
        pos.unmake_move(m, undo)
    return total
