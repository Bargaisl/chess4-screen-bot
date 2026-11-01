"""Тесты генерации ходов: perft, Zobrist, FEN, SAN."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chessvision.board import (Position, WHITE, BLACK, perft, perft_bulk,
                               START_FEN, infer_move)

PERFT_CASES = [
    (START_FEN, {1: 20, 2: 400, 3: 8902, 4: 197281}),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
     {1: 48, 2: 2039, 3: 97862}),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", {1: 14, 2: 191, 3: 2812, 4: 43238}),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
     {1: 6, 2: 264, 3: 9467}),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
     {1: 44, 2: 1486, 3: 62379}),
    ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
     {1: 46, 2: 2079, 3: 89890}),
]

fails = []


def check(name, cond, extra=""):
    print(("  OK   " if cond else "  FAIL ") + name + ("" if cond else "  " + str(extra)))
    if not cond:
        fails.append(name)


def test_perft():
    print("[perft]")
    for fen, expect in PERFT_CASES:
        pos = Position.from_fen(fen)
        for d, exp in sorted(expect.items()):
            got = perft_bulk(pos, d)
            check(f"d{d} {fen[:28]}", got == exp, f"got {got}, want {exp}")


def test_zobrist():
    print("[zobrist: инкрементальный ключ == полный пересчёт]")
    rnd = random.Random(42)
    ok = True
    for trial in range(30):
        pos = Position.start()
        for _ in range(rnd.randint(1, 60)):
            moves = pos.legal_moves()
            if not moves:
                break
            m = rnd.choice(moves)
            undo = pos.make_move(m)
            if pos.key != pos.compute_key():
                ok = False
                print(f"    рассинхрон после {pos.to_fen()}")
                break
            pos.unmake_move(m, undo)
            if pos.key != pos.compute_key():
                ok = False
                print("    ключ не восстановился после unmake")
                break
        if not ok:
            break
    check("инкрементальный Zobrist", ok)


def test_fen_roundtrip():
    print("[FEN]")
    fens = [START_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 1 5"]
    for fen in fens:
        p = Position.from_fen(fen)
        check(f"roundtrip {fen[:30]}", p.to_fen() == fen, p.to_fen())


def test_san():
    print("[SAN]")
    p = Position.start()
    seq = ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]
    want = ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"]
    for uci, exp in zip(seq, want):
        m = p.move_from_uci(uci)
        if m is None:
            check(f"uci {uci} легален", False)
            break
        got = p.san(m)
        check(f"san {uci} -> {got}", got == exp, f"want {exp}")
        p.make_move(m)


def test_infer():
    print("[поиск хода по расстановке]")
    p = Position.start()
    p2 = Position.start()
    m = p.move_from_uci("g1f3")
    p2.make_move(m)
    found = infer_move(p, p2.placement())
    check("hод g1f3 найден", found is not None and p.uci(found) == "g1f3",
          None if found is None else p.uci(found))

    # рокировка и взятие на проходе должны определяться однозначно
    p = Position.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    p2 = p.clone()
    p2.make_move(p.move_from_uci("e1g1"))
    found = infer_move(p, p2.placement())
    check("рокировка e1g1", found is not None and p.uci(found) == "e1g1")

    p = Position.from_fen("8/8/8/3pP3/8/8/8/4K2k w - d6 0 1")
    p2 = p.clone()
    p2.make_move(p.move_from_uci("e5d6"))
    found = infer_move(p, p2.placement())
    check("взятие на проходе e5d6", found is not None and p.uci(found) == "e5d6")


def test_mate():
    print("[мат/пат]")
    p = Position.from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    check("мат", p.game_status() == "checkmate", p.game_status())
    p = Position.from_fen("7k/5Q2/5K2/8/8/8/8/8 b - - 0 1")
    check("пат", p.game_status() == "stalemate", p.game_status())


if __name__ == "__main__":
    test_perft()
    test_zobrist()
    test_fen_roundtrip()
    test_san()
    test_infer()
    test_mate()
    print()
    if fails:
        print(f"ПРОВАЛЕНО: {len(fails)} -> {fails}")
        sys.exit(1)
    print("Все тесты доски пройдены.")
