import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess

from chessvision.board import Position, perft_bulk

FEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"

b = chess.Board(FEN)
p = Position.from_fen(FEN)
found = False
for mv in list(b.legal_moves):
    b.push(mv)
    m = p.move_from_uci(mv.uci())
    u = p.make_move(m)
    fresh = Position.from_fen(b.fen())
    f1, f2 = p.to_fen(), fresh.to_fen()
    c0 = perft_bulk(p, 2)              # на «живой» позиции, полученной make_move
    c1 = perft_bulk(Position.from_fen(f1), 2)
    c2 = perft_bulk(fresh, 2)
    if f1 != f2 or c1 != c2 or c0 != c2:
        found = True
        print("MISMATCH", mv.uci())
        print("  live  :", f1, "kings", p.kings, "ep", p.ep, "cast", p.castling, "count", c0)
        print("  made  :", f1, "kings", p.kings, "ep", p.ep, "cast", p.castling, "count", c1)
        print("  fresh:", f2, "kings", fresh.kings, "ep", fresh.ep, "cast", fresh.castling, "count", c2)
        break
    p.unmake_move(m, u)
    b.pop()
if not found:
    print("все дети совпадают")
