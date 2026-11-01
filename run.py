#!/usr/bin/env python3
"""Точка входа chess4-screen-bot."""
import sys
import os
import tkinter as tk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chessvision.app import App, _enable_dpi


def main():
    _enable_dpi()
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
