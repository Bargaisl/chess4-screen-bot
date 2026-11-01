"""Правдоподобие расстановки и определение того, чья сторона ходила."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chessvision.board import (BLACK, START_FEN, WHITE, expand_placement,
                               placement_issues, side_that_moved)

fails = []


def check(name, cond, extra=""):
    print(("  OK   " if cond else "  FAIL ") + name + ("" if cond else "  " + str(extra)))
    if not cond:
        fails.append(name)


START = START_FEN.split()[0]
AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"


def test_expand():
    print("[expand_placement]")
    grid = expand_placement(START)
    check("стартовая раскладка 8x8", grid is not None and len(grid) == 8
          and all(len(r) == 8 for r in grid))
    check("a8 = чёрная ладья", grid[0][0] == "r")
    check("e1 = белый король", grid[7][4] == "K")
    check("пустая клетка", grid[2][2] == "1")
    check("мусор -> None", expand_placement("not/a/fen/8/8/8/8/8") is None)
    check("7 строк -> None", expand_placement("8/8/8/8/8/8/8") is None)
    check("9 фигур в ряду -> None",
          expand_placement("PPPPPPPPP/8/8/8/8/8/8/K6k") is None)


def test_issues_ok():
    print("[placement_issues: нормальные позиции]")
    check("стартовая — без замечаний", placement_issues(START) == [],
          placement_issues(START))
    check("после 1.e4 — без замечаний", placement_issues(AFTER_E4) == [],
          placement_issues(AFTER_E4))
    check("рокировки/взятия — без замечаний",
          placement_issues("r3k2r/p1ppqp1b/2n1pn2/3N4/1p1P1P2/1P2P3/"
                           "P1P1Q1RB/R3K2R") == [])
    check("None -> ошибка", placement_issues(None) != [])


def test_issues_bad():
    print("[placement_issues: невозможные позиции]")
    no_king = "8/8/8/8/8/8/8/K7"
    got = placement_issues(no_king)
    check("нет чёрного короля", any("чёрных" in s for s in got), got)

    two_kings = "4k3/8/8/8/8/8/8/K3K3"
    got = placement_issues(two_kings)
    check("два белых короля", any("королей белых: 2" in s for s in got), got)

    pawn_back = "4k3/8/8/8/8/8/8/P3K3"
    got = placement_issues(pawn_back)
    check("пешка на 1-й горизонтали", any("1-й горизонтали" in s for s in got),
          got)

    pawn_top = "P3K3/8/8/8/8/8/8/4k3"
    got = placement_issues(pawn_top)
    check("пешка на 8-й горизонтали", any("8-й горизонтали" in s for s in got),
          got)

    many_pawns = "PPPPPPPP/PPPPPPPP/8/8/8/8/8/K6k"
    got = placement_issues(many_pawns)
    check("16 пешек", any("пешек белых" in s for s in got), got)

    check("мусор -> есть замечания", placement_issues("qwe/rtz/yui/8/8/8/8/8")
          != [])


def test_side_that_moved():
    print("[side_that_moved]")
    check("без изменений -> None", side_that_moved(START, START) is None)
    check("1.e4 -> ходили белые", side_that_moved(START, AFTER_E4) == WHITE)
    after_e5 = ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR")
    check("1.e4 e5 -> ходили чёрные",
          side_that_moved(AFTER_E4, after_e5) == BLACK)
    check("обе стороны пришли -> None",
          side_that_moved(START, after_e5) is None)
    check("кривая строка -> None", side_that_moved(START, "oops") is None)

    # рокировка: король и ладья «пришли» на новые клетки — ходили белые
    before = "r3k2r/8/8/8/8/8/8/R3K2R"
    after = "r3k2r/8/8/8/8/8/8/R4RK1"
    check("рокировка -> ходили белые", side_that_moved(before, after) == WHITE)

    # взятие: белая пешка заняла место чёрной — ходили белые
    before = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR"
    after = "rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR"
    check("взятие e4xd5 -> ходили белые", side_that_moved(before, after) == WHITE)

    # взятие чёрными: чёрная пешка пришла на место белой, белых «приходов» нет
    before2 = "rnbqkbnr/ppp1pppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR"
    after2 = "rnbqkbnr/ppp1pppp/8/8/3p4/8/PPP1PPPP/RNBQKBNR"
    check("взятие чёрными -> ходили чёрные",
          side_that_moved(before2, after2) == BLACK)
    check("обратное чтение (возврат) -> неясно",
          side_that_moved(after2, before2) is None)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    test_expand()
    test_issues_ok()
    test_issues_bad()
    test_side_that_moved()
    print()
    if fails:
        print(f"FAILURES: {len(fails)} -> {fails}")
        sys.exit(1)
    print("all sanity tests passed")
