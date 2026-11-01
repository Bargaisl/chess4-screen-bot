"""Бисекция расхождения perft: находит конкретный узел, где разъезжаются ходы."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess  # noqa

from chessvision.board import Position, perft_bulk


def my_perft(fen, depth):
    return perft_bulk(Position.from_fen(fen), depth)


def ref_perft(board, depth):
    if depth <= 0:
        return 1
    if depth == 1:
        return board.legal_moves.count()
    total = 0
    for mv in board.legal_moves:
        board.push(mv)
        total += ref_perft(board, depth - 1)
        board.pop()
    return total


def report(path, fen, d):
    a = my_perft(fen, d)
    b = ref_perft(chess.Board(fen), d)
    print(f"ПУТЬ: {path}\nFEN : {fen}\n  моя perft({d})={a}  эталон={b}")
    pos = Position.from_fen(fen)
    brd = chess.Board(fen)
    mine = {pos.uci(m) for m in pos.legal_moves()}
    ref = {m.uci() for m in brd.legal_moves}
    print("  лишние у меня :", sorted(mine - ref))
    print("  отсутствуют   :", sorted(ref - mine))


def bisect(board, path, depth):
    fen = board.fen()
    a, b = my_perft(fen, depth), ref_perft(board, depth)
    if a == b:
        return
    if depth <= 1:
        report(path, fen, max(depth, 1))
        return
    for mv in list(board.legal_moves):
        board.push(mv)
        bisect(board, path + " " + mv.uci(), depth - 1)
        board.pop()


def main():
    fen = sys.argv[1] if len(sys.argv) > 1 else \
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    # начинаем с детей, чтобы не считать всё дерево с корня
    board = chess.Board(fen)
    print("root:", my_perft(fen, depth), ref_perft(board, depth))
    for mv in list(board.legal_moves):
        board.push(mv)
        a, b = my_perft(board.fen(), depth - 1), ref_perft(board, depth - 1)
        if a != b:
            print(f"расхождение в дочернем узле {mv.uci()}: {a} vs {b}")
            bisect(board, mv.uci(), depth - 1)
        board.pop()


if __name__ == "__main__":
    main()
