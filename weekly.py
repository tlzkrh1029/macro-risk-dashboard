"""Fill the weekly record defined in DASHBOARD.md.

Usage:
    FRED_API_KEY=... python weekly.py                  # latest completed Friday
    FRED_API_KEY=... python weekly.py 2026-10-09       # a specific Friday
    FRED_API_KEY=... python weekly.py 2026-10-09 --save   # also write records/<date>.md

Requires: pandas, requests, xlrd
"""

import io
import os
import sys
import time
import unicodedata
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0"}
ACM_URL = "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xls"

RATES = [  # (label, source, id)
    ("2년물", "fred", "DGS2"),
    ("10년 실질", "fred", "DFII10"),
    ("10년 BE", "fred", "T10YIE"),
    ("기간프리미엄", "acm", "ACMTP10"),
]
SPREAD = [  # (label, source, id, unit, threshold % over 4 weeks)
    ("MOVE", "yahoo", "^MOVE", "", 20),
    ("HY OAS", "fred", "BAMLH0A0HYM2", "bp", 15),
    ("VIX", "yahoo", "^VIX", "", 30),
    ("DXY", "yahoo", "DX-Y.NYB", "", 3),
]


def fred(series_id):
    r = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={"series_id": series_id, "api_key": os.environ["FRED_API_KEY"], "file_type": "json",
                "observation_start": "2015-01-01"},
        timeout=60,
    )
    r.raise_for_status()
    obs = pd.DataFrame(r.json()["observations"])
    s = pd.to_numeric(obs["value"], errors="coerce")
    s.index = pd.to_datetime(obs["date"])
    return s.dropna()


def yahoo(ticker):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{requests.utils.quote(ticker)}"
           f"?period1={int(time.time()) - 400 * 86400}&period2={int(time.time())}&interval=1d")
    j = requests.get(url, headers=UA, timeout=60).json()["chart"]["result"][0]
    s = pd.Series(j["indicators"]["quote"][0]["close"],
                  index=pd.to_datetime(j["timestamp"], unit="s").normalize()).dropna()
    return s[~s.index.duplicated(keep="last")]


def acm(column):
    raw = requests.get(ACM_URL, headers=UA, timeout=120).content
    df = pd.read_excel(io.BytesIO(raw), sheet_name="ACM Daily")
    df["DATE"] = pd.to_datetime(df["DATE"], format="%d-%b-%Y")
    return df.set_index("DATE")[column].dropna()


def pad(text, width):
    """Left-align text, counting Korean characters as two columns."""
    cols = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(width - cols, 0)


def load(source, sid):
    return {"fred": fred, "yahoo": yahoo, "acm": acm}[source](sid)


def on_or_before(s, d):
    """Value on the Friday, or the week's last trading day if Friday had no value."""
    window = s.loc[d - timedelta(days=4):d]
    if window.empty:
        raise ValueError(f"no data in the week ending {d.date()}")
    return window.index[-1], window.iloc[-1]


def last_friday(today):
    """Most recent Friday whose US close is in; today's Friday counts only from Saturday."""
    return today - timedelta(days=(today.weekday() - 4) % 7 or 7)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    friday = pd.Timestamp(args[0]) if args else pd.Timestamp(last_friday(date.today()))
    if friday.weekday() != 4:
        sys.exit(f"{friday.date()} is not a Friday")
    prior = friday - timedelta(weeks=4)

    lines = [f"날짜: {friday.date()} (4주 전: {prior.date()})", "", "[금리]          값        4주 변화"]
    changes, notes = [], []
    for label, src, sid in RATES:
        s = load(src, sid)
        d1, v1 = on_or_before(s, friday)
        d0, v0 = on_or_before(s, prior)
        bp = (v1 - v0) * 100
        changes.append((label, bp))
        lines.append(f"{pad(label, 14)}{v1:>6.2f}%   {bp:+5.0f}bp")
        if d1 != friday or d0 != prior:
            notes.append(f"{label}: {d0.date()} → {d1.date()} 값 사용")
    top = sorted(changes, key=lambda x: abs(x[1]), reverse=True)[:2]
    lines.append(f"→ 원인: 1위 {top[0][0]} ({top[0][1]:+.0f}bp)  2위 {top[1][0]} ({top[1][1]:+.0f}bp)")

    lines += ["", "[전이]          값        4주 변화    기준 초과"]
    hits = 0
    for label, src, sid, unit, th in SPREAD:
        s = load(src, sid)
        d1, v1 = on_or_before(s, friday)
        d0, v0 = on_or_before(s, prior)
        pct = (v1 / v0 - 1) * 100
        over = pct >= th
        hits += over
        shown = f"{v1 * 100:.0f}bp" if unit == "bp" else f"{v1:.2f}"
        lines.append(f"{pad(label, 14)}{shown:>7}   {pct:+6.1f}%    {'■' if over else '□'} (기준 +{th}%)")
        if d1 != friday or d0 != prior:
            notes.append(f"{label}: {d0.date()} → {d1.date()} 값 사용")
    verdict = "안정" if hits == 0 else ("경계" if hits <= 2 else "스트레스")
    lines.append(f"→ 판정: {verdict} ({hits}/4)")
    if notes:
        lines += ["", "데이터 메모: " + "; ".join(notes)]

    text = "\n".join(lines)
    print(text)
    if "--save" in sys.argv:
        out = Path(__file__).resolve().parent / "records" / f"{friday.date()}.md"
        out.parent.mkdir(exist_ok=True)
        out.write_text(f"# 주간 기록 {friday.date()}\n\n```\n{text}\n```\n\n메모 한 줄: \n")
        print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
