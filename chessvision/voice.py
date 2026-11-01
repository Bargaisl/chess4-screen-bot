"""Озвучка ходов в реальном времени через Windows Speech API (SAPI)."""
from __future__ import annotations

import threading


def san_to_voice(san: str) -> str:
    """Преобразование шахматной нотации (SAN) в естественную русскую речь."""
    if not san:
        return ""
    san = san.strip()
    if san in ("O-O", "0-0"):
        return "короткая рокировка"
    if san in ("O-O-O", "0-0-0"):
        return "длинная рокировка"

    pieces = {
        "N": "конь ",
        "B": "слон ",
        "R": "ладья ",
        "Q": "ферзь ",
        "K": "король ",
    }
    letters = {
        "a": "а ",
        "b": "бэ ",
        "c": "цэ ",
        "d": "дэ ",
        "e": "е ",
        "f": "эф ",
        "g": "жэ ",
        "h": "аш ",
    }

    res = ""
    if san and san[0] in pieces:
        res += pieces[san[0]]
        san = san[1:]

    for ch in san:
        if ch in letters:
            res += letters[ch]
        elif ch.isdigit():
            res += ch + " "
        elif ch == "x":
            res += "бьет "
        elif ch == "+":
            res += "шах "
        elif ch == "#":
            res += "мат "
        elif ch == "=":
            res += "превращение "

    return res.strip()


import queue


class VoiceAnnouncer:
    """Асинхронный диктор шахматных ходов через выделенный COM-поток SAPI."""

    def __init__(self):
        self._q = queue.Queue()
        self._available = False
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._ready.wait(0.30)

    def _worker(self):
        try:
            import pythoncom
            import win32com.client
            pythoncom.CoInitialize()
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            self._available = True
            self._ready.set()

            # Автоматически выбираем русский голос (Microsoft Irina), если он установлен
            voices = speaker.GetVoices()
            for i in range(voices.Count):
                v = voices.Item(i)
                desc = v.GetDescription().lower()
                if "russian" in desc or "irina" in desc:
                    speaker.Voice = v
                    break

            while True:
                text = self._q.get()
                if text is None:
                    break
                try:
                    # 0 = синхронно внутри фонового воркера (не блокирует GUI)
                    speaker.Speak(text, 0)
                except Exception:
                    pass
                self._q.task_done()
            pythoncom.CoUninitialize()
        except Exception:
            self._available = False

    def speak(self, text: str):
        """Озвучивание текста в фоновом потоке без блокировки GUI."""
        if not text:
            return
        # Очищаем очередь от устаревших фраз, чтобы озвучивать только самый актуальный ход
        while not self._q.empty():
            try:
                self._q.get_nowait()
                self._q.task_done()
            except Exception:
                break
        self._q.put(text)

    def speak_move(self, san: str):
        spoken = san_to_voice(san)
        if spoken:
            self.speak(spoken)
