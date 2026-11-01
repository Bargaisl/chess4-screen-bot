"""Модуль человекоподобного движения мыши на основе сплайнов Безье и кинематики руки."""
from __future__ import annotations

import math
import random
import time
import pyautogui

pyautogui.FAILSAFE = False


def _min_jerk(t: float) -> float:
    """Кинематический профиль скорости движения руки человека (Flash & Hogan).
    
    Обеспечивает плавное ускорение от нуля, крейсерскую скорость в середине
    и мягкое естественное торможение перед целевой точкой.
    """
    t = max(0.0, min(1.0, t))
    return 10.0 * (t ** 3) - 15.0 * (t ** 4) + 6.0 * (t ** 5)


def _cubic_bezier(p0: tuple[float, float],
                  p1: tuple[float, float],
                  p2: tuple[float, float],
                  p3: tuple[float, float],
                  t: float) -> tuple[float, float]:
    """Точка на кубической кривой Безье в момент времени t in [0, 1]."""
    u = 1.0 - t
    tt = t * t
    uu = u * u
    x = uu * u * p0[0] + 3.0 * uu * t * p1[0] + 3.0 * u * tt * p2[0] + tt * t * p3[0]
    y = uu * u * p0[1] + 3.0 * uu * t * p1[1] + 3.0 * u * tt * p2[1] + tt * t * p3[1]
    return x, y


def human_move(tx: float, ty: float, duration: float | None = None):
    """Плавное перемещение курсора к цели по естественной человеческой траектории."""
    start_x, start_y = pyautogui.position()
    dx = tx - start_x
    dy = ty - start_y
    dist = math.hypot(dx, dy)

    if dist < 4.0:
        pyautogui.moveTo(tx, ty)
        return

    if duration is None:
        # Человеческое время перемещения по закону Фиттса: от 120мс на близкие до 260мс на дальние
        duration = random.uniform(0.12, 0.22) + (dist / 1400.0) * 0.12

    # Вычисляем перпендикулярный вектор для дуги замаха кисти/локтя
    nx = -dy / dist
    ny = dx / dist

    # Случайное боковое отклонение дуги (до 15-20% от дистанции)
    side = random.choice([-1.0, 1.0])
    dev1 = dist * random.uniform(0.08, 0.18) * side
    dev2 = dist * random.uniform(-0.06, 0.12) * side

    p0 = (float(start_x), float(start_y))
    p1 = (start_x + dx * 0.32 + nx * dev1, start_y + dy * 0.32 + ny * dev1)
    p2 = (start_x + dx * 0.68 + nx * dev2, start_y + dy * 0.68 + ny * dev2)
    p3 = (float(tx), float(ty))

    steps = max(12, int(duration * 90.0))
    dt = duration / float(steps)

    for i in range(1, steps + 1):
        t_raw = float(i) / float(steps)
        t_curved = _min_jerk(t_raw)
        cx, cy = _cubic_bezier(p0, p1, p2, p3, t_curved)

        # Микро-тремор кисти (до 0.8 пикселя)
        if i < steps:
            cx += random.uniform(-0.6, 0.6)
            cy += random.uniform(-0.6, 0.6)

        pyautogui.moveTo(cx, cy)
        time.sleep(dt)

    pyautogui.moveTo(tx, ty)


def human_click():
    """Естественный клик с анатомической задержкой нажатия клавиши (40-90 мс)."""
    pyautogui.mouseDown()
    time.sleep(random.uniform(0.045, 0.085))
    pyautogui.mouseUp()


def human_move_and_click(fx: float, fy: float, tx: float, ty: float):
    """Полный цикл выполнения шахматного хода: клик исходной клетки -> клик целевой."""
    # Небольшой случайный офсет внутри клетки (+-3 px), чтобы не бить в один и тот же пиксель
    from_x = fx + random.uniform(-3.5, 3.5)
    from_y = fy + random.uniform(-3.5, 3.5)
    to_x = tx + random.uniform(-3.5, 3.5)
    to_y = ty + random.uniform(-3.5, 3.5)

    # 1. Перемещение к начальной фигуре
    human_move(from_x, from_y)
    time.sleep(random.uniform(0.03, 0.08))
    human_click()

    # 2. Небольшая человеческая пауза между взятием фигуры и постановкой
    time.sleep(random.uniform(0.09, 0.18))

    # 3. Перемещение к целевой клетке с возможным микро-овершутом
    if random.random() < 0.25:
        # Лёгкий занос на 2-4 пикселя и быстрая доводка (overshoot)
        over_x = to_x + random.uniform(-3.0, 3.0)
        over_y = to_y + random.uniform(-3.0, 3.0)
        human_move(over_x, over_y)
        time.sleep(random.uniform(0.02, 0.05))
        human_move(to_x, to_y, duration=0.04)
    else:
        human_move(to_x, to_y)

    time.sleep(random.uniform(0.03, 0.07))
    human_click()


def human_drag_and_drop(fx: float, fy: float, tx: float, ty: float):
    """Выполнение хода перетаскиванием фигуры (Drag-and-Drop) по кривым Безье."""
    from_x = fx + random.uniform(-3.5, 3.5)
    from_y = fy + random.uniform(-3.5, 3.5)
    to_x = tx + random.uniform(-3.5, 3.5)
    to_y = ty + random.uniform(-3.5, 3.5)

    # 1. Плавный подвод курсора к начальной фигуре
    human_move(from_x, from_y)
    time.sleep(random.uniform(0.04, 0.08))

    # 2. Захват фигуры (зажатие левой кнопки мыши)
    pyautogui.mouseDown()
    time.sleep(random.uniform(0.05, 0.10))

    # 3. Перетаскивание зажатой фигуры по человеческой траектории к целевой клетке
    if random.random() < 0.20:
        over_x = to_x + random.uniform(-2.5, 2.5)
        over_y = to_y + random.uniform(-2.5, 2.5)
        human_move(over_x, over_y)
        time.sleep(random.uniform(0.02, 0.04))
        human_move(to_x, to_y, duration=0.04)
    else:
        human_move(to_x, to_y)

    time.sleep(random.uniform(0.04, 0.09))

    # 4. Отпускание фигуры на целевой клетке
    pyautogui.mouseUp()
