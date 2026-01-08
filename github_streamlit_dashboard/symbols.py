from __future__ import annotations

import re
from typing import List


def parse_symbol_list(raw: str) -> List[str]:
    if not raw:
        return []
    parts = re.split(r"[\n,;]+", raw)
    symbols = []
    seen = set()
    for p in parts:
        s = p.strip().upper()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        symbols.append(s)
    return symbols


def format_set_symbol(symbol: str) -> str:
    s = (symbol or "").strip().upper()
    if not s:
        return s
    if ":" in s:
        return s
    return f"SET:{s}"


def short_symbol(symbol: str) -> str:
    s = (symbol or "").strip()
    if ":" in s:
        return s.split(":", 1)[1]
    return s
