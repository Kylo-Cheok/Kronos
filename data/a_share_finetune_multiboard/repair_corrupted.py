# -*- coding: utf-8 -*-
"""Repair CSVs whose TDX MCP rows carry garbage OHLC values (negative / ~0 prices).

The TDX MCP responses themselves contained corrupted rows (e.g. 000002
2010-08-13: open=-0.24, close=0.14 while volume/amount are real), so no local
fix can recover those bars.  These symbols are re-fetched wholesale from the
akshare sina source (same pipeline as ``fetch_missing_20.py``), replacing the
CSV entirely so each symbol stays internally consistent qfq.

Run from the repository root::

    python data/a_share_finetune_multiboard/repair_corrupted.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from fetch_missing_20 import CSV_DIR, fetch_sina, save_safely, to_canonical  # noqa: E402

# Symbols whose CSVs contain rows with open/high/low/close <= 0 (garbage values
# leaked by the TDX MCP server).  Detected by scanning every CSV for
# non-positive OHLC; all 34 confirmed against the raw TDX responses.
CORRUPTED: list[tuple[str, str]] = [
    ("000001", "平安银行"),
    ("000157", "中联重科"),
    ("000333", "美的集团"),
    ("000538", "云南白药"),
    ("002241", "歌尔股份"),
    ("002415", "海康威视"),
    ("002466", "天齐锂业"),
    ("002493", "荣盛石化"),
    ("002594", "比亚迪"),
    ("300003", "乐普医疗"),
    ("300059", "东方财富"),
    ("300122", "智飞生物"),
    ("300750", "宁德时代"),
    ("600031", "三一重工"),
    ("600276", "恒瑞医药"),
    ("600298", "安琪酵母"),
    ("600760", "中航沈飞"),
    ("601012", "隆基绿能"),
    ("601238", "广汽集团"),
    ("601899", "紫金矿业"),
    ("603259", "药明康德"),
    ("603288", "海天味业"),
    ("688169", "石头科技"),
]


def sina_prefix(code: str) -> str:
    return "sh" if code.startswith("6") else "sz"


def main() -> int:
    ok, failed = [], []
    for code, name in CORRUPTED:
        print(f"{code} {name} ...", flush=True)
        df = fetch_sina(code, sina_prefix(code))
        if df is None:
            failed.append((code, name, "fetch failed"))
            continue
        try:
            frame = to_canonical(df)
        except ValueError as exc:
            failed.append((code, name, str(exc)))
            continue
        if save_safely(frame, code):
            ok.append((code, name, len(frame), frame["timestamps"].iloc[0], frame["timestamps"].iloc[-1]))
        else:
            failed.append((code, name, "too few rows"))
        time.sleep(1)

    print("\n" + "=" * 72)
    print("REPAIR SUMMARY")
    print("=" * 72)
    for code, name, n, first, last in ok:
        print(f"  OK   {code} {name:<6} {n:>5} rows  {first} -> {last}")
    for code, name, reason in failed:
        print(f"  FAIL {code} {name:<6} {reason}")
    print(f"\n{len(ok)} repaired, {len(failed)} failed")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
