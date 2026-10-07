"""Tiny ANSI color helpers for log messages."""


def _wrap(code: str):
    return lambda text: f"\033[{code}m{text}\033[0m"


red = _wrap("91")
green = _wrap("32")
yellow = _wrap("33")
