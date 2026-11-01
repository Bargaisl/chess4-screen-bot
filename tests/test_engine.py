"""Тесты движка: мат в 1, выигрыш материала, сохранность позиции, скорость."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess  # noqa: эталон

from chessvision.board import Position
from chessvision.engine import Engine, TEMPO, evaluate

fails = []


def check(name, cond, extra=""):
    print(("  OK   " if cond else "  FAIL ") + name + ("" if cond else "  " + str(extra)))
    if not cond:
        fails.append(name)


MATE_IN_1 = [
    ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "мат ладьёй по 8-й"),
    ("7k/8/3Q2K1/8/8/8/8/8 w - - 0 1", "ферзь d6 -> d8"),
    ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1",
     "ферзь на f7 (учёная матовая)"),
    ("6k1/5ppp/8/8/8/8/8/3Q2K1 w - - 0 1", "ферзь d1 на d8"),
]


def test_eval_symmetry():
    print("[оценка]")
    p = Position.start()
    check("стартовая позиция симметрична (+темп)", evaluate(p) == TEMPO,
          evaluate(p))
    p = Position.from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    check("лишний ферзь у белых > 500", evaluate(p) > 500, evaluate(p))
    p2 = Position.from_fen("4k3/8/8/8/8/8/8/3QK3 b - - 0 1")
    check("то же глазами чёрных < -500", evaluate(p2) < -500, evaluate(p2))


def test_mate_in_one():
    print("[мат в 1 ход]")
    eng = Engine(prefer_builtin=True, use_book=False)
    for fen, name in MATE_IN_1:
        pos = Position.from_fen(fen)
        res = eng.analyze(pos, movetime_ms=800)
        mv = res["move"]
        if mv is None:
            check(name, False, "ход не найден")
            continue
        board = chess.Board(fen)
        board.push(chess.Move.from_uci(pos.uci(mv)))
        check(f"{name}: {res['san']}", board.is_checkmate(),
              f"не мат (счёт {res['score']}, глубина {res['depth']})")
    eng.close()


def test_win_material():
    print("[выигрыш висячей ладьи]")
    fen = "r6k/8/8/8/8/8/8/R5K1 w - - 0 1"
    eng = Engine(prefer_builtin=True, use_book=False)
    pos = Position.from_fen(fen)
    res = eng.analyze(pos, movetime_ms=800)
    check(f"ладья выиграна ({res['uci']})", res["score"] > 200, res["score"])
    check("походил ладьёй a1", res["uci"] == "a1a8", res["uci"])
    eng.close()


def test_position_intact():
    print("[позиция не портится после поиска]")
    eng = Engine(prefer_builtin=True, use_book=False)
    fens = ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1")
    for fen in fens:
        pos = Position.from_fen(fen)
        before = pos.to_fen()
        res = eng.analyze(pos, movetime_ms=500)
        legal = {pos.uci(m) for m in pos.legal_moves()}
        check(f"ход легален: {res['uci']}", res["uci"] in legal, res["uci"])
        check(f"позиция сохранена: {fen[:22]}", pos.to_fen() == before,
              pos.to_fen())
    eng.close()


def test_game_over():
    print("[конец партии]")
    eng = Engine(prefer_builtin=True, use_book=False)
    p = Position.from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    res = eng.analyze(p, movetime_ms=300)
    check("мат в позиции -> хода нет", res["move"] is None, res)
    eng.close()


def test_self_play():
    print("[самоигра 10 полуходов без ошибок]")
    eng = Engine(prefer_builtin=True, use_book=False)
    pos = Position.start()
    board = chess.Board(pos.to_fen())
    ok = True
    for i in range(10):
        res = eng.analyze(pos, movetime_ms=250)
        mv = res["move"]
        if mv is None or res["uci"] not in {pos.uci(m) for m in pos.legal_moves()}:
            ok = False
            print(f"    плохой ход на полуходе {i}: {res}")
            break
        pos.make_move(mv)
        board.push(chess.Move.from_uci(res["uci"]))
    check("10 полуходов сыграно", ok)
    check("расстановка соответствует эталону",
          board.board_fen() == pos.placement(), f"{board.board_fen()} vs {pos.placement()}")
    eng.close()


def test_speed():
    print("[скорость и глубина]")
    eng = Engine(prefer_builtin=True, use_book=False)
    pos = Position.start()
    res = eng.analyze(pos, movetime_ms=3000)
    check(f"глубина >= 4 за 3 с (факт {res['depth']})", res["depth"] >= 4)
    print(f"       узлов={res['nodes']} nps={res['nps']} "
          f"ход={res['uci']} pv={' '.join(res['pv'][:8])}")
    check("первый ход разумный",
          res["uci"] in {"e2e4", "d2d4", "c2c4", "g1f3", "b1c3"}, res["uci"])
    eng.close()


def test_book():
    print("[дебютная книга]")
    eng = Engine(prefer_builtin=True, use_book=True)
    pos = Position.start()
    res = eng.analyze(pos, movetime_ms=2000)
    check(f"на старте книга отвечает мгновенно ({res['uci']})",
          res.get("book") is True and
          res["uci"] in {"e2e4", "d2d4", "c2c4", "g1f3"}, res)
    # ход белых e4, ход чёрных из книги
    pos.make_move(pos.move_from_uci(res["uci"]))
    res2 = eng.analyze(pos, movetime_ms=2000)
    check(f"ответ чёрных из книги ({res2['uci']})", res2.get("book") is True,
          res2)
    eng.close()


if __name__ == "__main__":
    test_eval_symmetry()
    test_mate_in_one()
    test_win_material()
    test_position_intact()
    test_game_over()
    test_self_play()
    test_speed()
    test_book()
    print()
    if fails:
        print(f"ПРОВАЛЕНО: {len(fails)} -> {fails}")
        sys.exit(1)
    print("Все тесты движка пройдены.")
