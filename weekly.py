"""Fill the weekly record defined in DASHBOARD.md, and optionally build the dashboard page.

Usage:
    FRED_API_KEY=... python weekly.py                    # latest completed Friday
    FRED_API_KEY=... python weekly.py 2026-10-09         # a specific Friday
    FRED_API_KEY=... python weekly.py --save             # also write records/<date>.md
    FRED_API_KEY=... python weekly.py --page             # also write page/dashboard.html

Requires: pandas, requests, xlrd
"""

import io
import json
import os
import re
import sys
import time
import unicodedata
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
UA = {"User-Agent": "Mozilla/5.0"}
ACM_URL = "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xls"
HISTORY_WEEKS = 12   # verdict strip on the page
SPARK_WEEKS = 13     # weekly points per sparkline

RATES = [  # (key, label, what it shows, source, id)
    ("dgs2", "2년물", "연준 정책 기대", "fred", "DGS2"),
    ("dfii10", "10년 실질", "성장과 긴축이 반영된 실질 비용", "fred", "DFII10"),
    ("t10yie", "10년 BE", "인플레이션 기대", "fred", "T10YIE"),
    ("acmtp10", "기간프리미엄", "재정·공급·불확실성", "acm", "ACMTP10"),
]
SPREAD = [  # (key, label, what it shows, source, id, unit, threshold % over 4 weeks)
    ("move", "MOVE", "금리 변동성", "yahoo", "^MOVE", "", 20),
    ("hy", "HY OAS", "신용", "fred", "BAMLH0A0HYM2", "bp", 15),
    ("vix", "VIX", "주식 변동성", "yahoo", "^VIX", "", 30),
    ("dxy", "DXY", "글로벌 유동성", "yahoo", "DX-Y.NYB", "", 3),
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


def load_all():
    loaders = {"fred": fred, "yahoo": yahoo, "acm": acm}
    return {row[0]: loaders[row[3]](row[4]) for row in RATES + SPREAD}


def on_or_before(s, d):
    """Value on the Friday, or the week's last trading day if Friday had no value."""
    window = s.loc[d - timedelta(days=4):d]
    if window.empty:
        raise ValueError(f"no data in the week ending {d.date()}")
    return window.index[-1], float(window.iloc[-1])


def last_friday(today):
    """Most recent Friday whose US close is in; today's Friday counts only from Saturday."""
    return today - timedelta(days=(today.weekday() - 4) % 7 or 7)


def verdict_of(hits):
    return "안정" if hits == 0 else ("경계" if hits <= 2 else "스트레스")


def compute(series, friday):
    prior = friday - timedelta(weeks=4)
    rec = {"friday": str(friday.date()), "prior": str(prior.date()), "rates": [], "spread": [], "notes": []}
    for key, label, what, _, sid in RATES:
        d1, v1 = on_or_before(series[key], friday)
        d0, v0 = on_or_before(series[key], prior)
        rec["rates"].append({"key": key, "label": label, "what": what, "id": sid,
                             "value": v1, "change_bp": (v1 - v0) * 100})
        if d1 != friday or d0 != prior:
            rec["notes"].append(f"{label}: {d0.date()} → {d1.date()} 값 사용")
    ranked = sorted(rec["rates"], key=lambda r: abs(r["change_bp"]), reverse=True)
    for i, r in enumerate(ranked):
        r["rank"] = i + 1
    for key, label, what, _, sid, unit, th in SPREAD:
        d1, v1 = on_or_before(series[key], friday)
        d0, v0 = on_or_before(series[key], prior)
        pct = (v1 / v0 - 1) * 100
        rec["spread"].append({"key": key, "label": label, "what": what, "id": sid, "unit": unit,
                              "value": v1 * 100 if unit == "bp" else v1, "change_pct": pct,
                              "threshold": th, "over": pct >= th})
        if d1 != friday or d0 != prior:
            rec["notes"].append(f"{label}: {d0.date()} → {d1.date()} 값 사용")
    rec["hits"] = sum(r["over"] for r in rec["spread"])
    rec["verdict"] = verdict_of(rec["hits"])
    rec["cause"] = [ranked[0]["label"], ranked[1]["label"]]
    return rec


def pad(text, width):
    """Left-align text, counting Korean characters as two columns."""
    cols = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(width - cols, 0)


def render_text(rec):
    lines = [f"날짜: {rec['friday']} (4주 전: {rec['prior']})", "", "[금리]          값        4주 변화"]
    for r in rec["rates"]:
        lines.append(f"{pad(r['label'], 14)}{r['value']:>6.2f}%   {r['change_bp']:+5.0f}bp")
    top = sorted(rec["rates"], key=lambda r: r["rank"])[:2]
    lines.append(f"→ 원인: 1위 {top[0]['label']} ({top[0]['change_bp']:+.0f}bp)  "
                 f"2위 {top[1]['label']} ({top[1]['change_bp']:+.0f}bp)")
    lines += ["", "[전이]          값        4주 변화    기준 초과"]
    for r in rec["spread"]:
        shown = f"{r['value']:.0f}bp" if r["unit"] == "bp" else f"{r['value']:.2f}"
        lines.append(f"{pad(r['label'], 14)}{shown:>7}   {r['change_pct']:+6.1f}%    "
                     f"{'■' if r['over'] else '□'} (기준 +{r['threshold']}%)")
    lines.append(f"→ 판정: {rec['verdict']} ({rec['hits']}/4)")
    if rec["notes"]:
        lines += ["", "데이터 메모: " + "; ".join(rec["notes"])]
    return "\n".join(lines)


def record_path(friday):
    return ROOT / "records" / f"{friday.date()}.md"


def read_memo(friday):
    path = record_path(friday)
    if not path.exists():
        return ""
    m = re.search(r"^메모 한 줄:[ \t]*(.*)$", path.read_text(), re.M)
    return m.group(1).strip() if m else ""


def save_record(rec, friday):
    path = record_path(friday)
    memo = read_memo(friday)  # keep a memo already written by hand
    path.parent.mkdir(exist_ok=True)
    path.write_text(f"# 주간 기록 {rec['friday']}\n\n```\n{render_text(rec)}\n```\n\n메모 한 줄: {memo}\n")
    return path


def build_page(series, rec, friday):
    fridays = [friday - timedelta(weeks=k) for k in range(SPARK_WEEKS - 1, -1, -1)]
    spark = {}
    for key, *_rest in RATES + SPREAD:
        unit_bp = any(row[0] == key and row[5] == "bp" for row in SPREAD)
        spark[key] = [{"d": str(f.date()), "v": on_or_before(series[key], f)[1] * (100 if unit_bp else 1)}
                      for f in fridays]
    history = []
    for k in range(HISTORY_WEEKS - 1, -1, -1):
        f = friday - timedelta(weeks=k)
        h = compute(series, f)
        history.append({"d": h["friday"], "hits": h["hits"], "verdict": h["verdict"],
                        "over": [r["label"] for r in h["spread"] if r["over"]], "cause": h["cause"]})
    data = dict(rec, spark=spark, history=history, memo=read_memo(friday))
    template = (ROOT / "page" / "template.html").read_text()
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    out = ROOT / "page" / "dashboard.html"
    out.write_text(template.replace("/*__DATA__*/null", payload))
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    friday = pd.Timestamp(args[0]) if args else pd.Timestamp(last_friday(date.today()))
    if friday.weekday() != 4:
        sys.exit(f"{friday.date()} is not a Friday")
    series = load_all()
    rec = compute(series, friday)
    print(render_text(rec))
    if "--save" in sys.argv:
        print(f"\nsaved {save_record(rec, friday)}")
    if "--page" in sys.argv:
        print(f"built {build_page(series, rec, friday)}")


if __name__ == "__main__":
    main()
