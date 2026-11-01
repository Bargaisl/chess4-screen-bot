"""Рекурсивный дифф генерации ходов против python-chess (эталон)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess  # noqa: эталон

from chessvision.board import Position

FENS = [
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 b kq - 0 1",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "8/PPPk4/8/8/8/8/4Kppp/8 w - - 0 1",
    "8/PPPk4/8/8/8/8/4Kppp/8 b - - 0 1",
    "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "8/5k2/8/8/8/8/2K5/8 w - - 0 1",
    "8/8/8/3k4/8/8/3K4/8 b - - 0 1",
]

problems = []


def mine_moves(fen):
    p = Position.from_fen(fen)
    return {p.uci(m) for m in p.legal_moves()}


def ref_moves(fen):
    b = chess.Board(fen)
    return {m.uci() for m in b.legal_moves}


def walk(board, path, depth):
    fen = board.fen()
    a, c = mine_moves(fen), {m.uci() for m in board.legal_moves}
    if a != c:
        problems.append((path or "(старт)", fen, sorted(a - c), sorted(c - a)))
        return
    if depth <= 0:
        return
    for mv in list(board.legal_moves):
        board.push(mv)
        walk(board, path + " " + mv.uci(), depth - 1)
        board.pop()


def main():
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    used = 0
    for fen in FENS:
        board = chess.Board(fen)
        if not board.is_valid():
            print(f"  пропущена нелегальная позиция: {fen}")
            continue
        used += 1
        walk(board, "", depth)
    print(f"проверено позиций: {used}")
    if problems:
        print(f"расхождений: {len(problems)}")
        for path, fen, extra, missing in problems[:20]:
            print(f"  путь: {path}")
            print(f"  фen : {fen}")
            if extra:
                print(f"    лишние у меня : {extra}")
            if missing:
                print(f"    отсутствуют   : {missing}")
        sys.exit(1)
    print(f"все позиции согласуются с эталоном (глубина {depth})")


if __name__ == "__main__":
    main()
