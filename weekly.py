"""Fill the weekly record defined in DASHBOARD.md, and optionally build the dashboard page.

Usage:
    FRED_API_KEY=... python weekly.py                    # latest completed Friday
    FRED_API_KEY=... python weekly.py 2026-10-09         # a specific Friday
    FRED_API_KEY=... python weekly.py --save             # also write records/<date>.md
    FRED_API_KEY=... python weekly.py --page             # also write page/dashboard.html
    FRED_API_KEY=... python weekly.py --latest --page    # page for the latest US close (the page's refresh button)

--latest picks the latest US trading day whose close is in. On a Friday that is the weekly page; on any other
day it is a mid-week provisional page, which compares that close with the same weekday 4 weeks earlier and is
never saved as a weekly record.

Requires: pandas, requests, xlrd
"""

import io
import json
import math
import os
import re
import sys
import time
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
UA = {"User-Agent": "Mozilla/5.0"}
ACM_URL = "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xls"
REFRESH_TRIGGER = "trig_01D7QMaWrEGmARRpbm7YKDZo"  # poke-only scheduled task behind the page's refresh buttons
CLOSE_HOUR_ET = 17   # a weekday's close counts from 17:00 New York time; Yahoo serves the unfinished bar before that
HISTORY_WEEKS = 12   # verdict strip on the page
SPARK_WEEKS = 13     # weekly points per sparkline
YAHOO_DAYS = 1300    # enough to recompute the rule back to when FRED's HY OAS history starts
REGIME_BP = 15       # 10y 4-week move that counts as a direction for the regime label
SCORE_WEEKS = 8      # scorecard: what happened in the 8 weeks after each verdict
SCORE_DD = -5        # S&P 500 max drawdown (%) that counts as a fall
SCORE_HY = 50        # HY OAS max widening (bp) that counts as credit stress

RATES = [  # (key, label, what it shows, source, id)
    ("dgs2", "2년물", "연준 정책 기대", "fred", "DGS2"),
    ("dfii10", "10년 실질", "성장과 긴축이 반영된 실질 비용", "fred", "DFII10"),
    ("t10yie", "10년 BE", "인플레이션 기대", "fred", "T10YIE"),
    ("acmtp10", "기간프리미엄", "재정·공급·불확실성", "acm", "ACMTP10"),
]
TOTAL = ("dgs10", "10년물", "10년 실질과 10년 BE를 더한 값", "fred", "DGS10")  # reference line, not ranked
SPREAD = [  # (key, label, what it shows, source, id, unit, threshold % over 4 weeks)
    ("move", "MOVE", "금리 변동성", "yahoo", "^MOVE", "", 20),
    ("hy", "HY OAS", "신용", "fred", "BAMLH0A0HYM2", "bp", 15),
    ("vix", "VIX", "주식 변동성", "yahoo", "^VIX", "", 30),
    ("dxy", "DXY", "글로벌 유동성", "yahoo", "DX-Y.NYB", "", 3),
]
OUTCOME = ("spx", "S&P 500", "", "yahoo", "^GSPC")  # scorecard only, not an indicator
PLAIN = {"move": "채권 변동성", "hy": "신용", "vix": "주식 변동성", "dxy": "달러"}
REGIMES = {  # (10y direction, any transmission indicator over?) -> (name, meaning)
    ("상승", True): ("금리발 전이", "10년 금리가 오르면서 그 충격이 위험자산으로 번지고 있습니다."),
    ("횡보", True): ("금리 외 요인", "10년 금리는 크게 움직이지 않았는데 위험자산 쪽 지표가 기준을 넘었습니다."),
    ("하락", True): ("금리 하락 속 전이", "10년 금리가 내리는 가운데 위험자산 쪽 지표가 기준을 넘었습니다. 금리 상승이 번진 것은 아닙니다."),
    ("상승", False): ("질서 있는 금리 상승", "10년 금리는 올랐지만 위험자산 쪽 지표는 모두 기준 아래입니다."),
    ("횡보", False): ("큰 변화 없음", "10년 금리는 크게 움직이지 않았고, 위험자산 쪽 지표도 모두 기준 아래입니다."),
    ("하락", False): ("질서 있는 금리 하락", "10년 금리는 내렸고, 위험자산 쪽 지표는 모두 기준 아래입니다."),
}


def retry(fetch, *args, tries=3, wait=20):
    """Retry a fetch; a FRED maintenance window answers with an HTML page, which fails to parse as JSON."""
    for i in range(tries):
        try:
            return fetch(*args)
        except (requests.RequestException, ValueError, KeyError, TypeError, IndexError):
            if i == tries - 1:
                raise
            time.sleep(wait * (i + 1))


def _fred(series_id):
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


def _yahoo(ticker):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{requests.utils.quote(ticker)}"
           f"?period1={int(time.time()) - YAHOO_DAYS * 86400}&period2={int(time.time())}&interval=1d")
    j = requests.get(url, headers=UA, timeout=60).json()["chart"]["result"][0]
    s = pd.Series(j["indicators"]["quote"][0]["close"],
                  index=pd.to_datetime(j["timestamp"], unit="s").normalize()).dropna()
    return s[~s.index.duplicated(keep="last")]


def fred(series_id, tries=3):
    """FRED series with retries; a failure is re-raised without the request URL, which carries the API key."""
    try:
        return retry(_fred, series_id, tries=tries)
    except requests.RequestException as e:
        status = getattr(e.response, "status_code", "") or ""
        raise RuntimeError(f"FRED {series_id}: {type(e).__name__} {status}".strip()) from None


def yahoo(ticker, tries=3):
    return retry(_yahoo, ticker, tries=tries)


def acm(column):
    raw = requests.get(ACM_URL, headers=UA, timeout=120).content
    df = pd.read_excel(io.BytesIO(raw), sheet_name="ACM Daily")
    df["DATE"] = pd.to_datetime(df["DATE"], format="%d-%b-%Y")
    return df.set_index("DATE")[column].dropna()


def load_all():
    loaders = {"fred": fred, "yahoo": yahoo, "acm": acm}
    series = {row[0]: loaders[row[3]](row[4]) for row in RATES + SPREAD + [TOTAL]}
    try:  # the scorecard is optional; a failed S&P fetch must not stop the weekly record
        series[OUTCOME[0]] = loaders[OUTCOME[3]](OUTCOME[4])
    except Exception as e:  # noqa: BLE001
        print(f"warning: {OUTCOME[4]} not loaded, scorecard skipped ({e})", file=sys.stderr)
        series[OUTCOME[0]] = None
    spx = series[OUTCOME[0]]
    if spx is not None:  # Cboe quotes VIX on Memorial Day and Labor Day; such a quote is no close to compare with
        v = series["vix"]
        series["vix"] = v[v.index.isin(spx.index) | (v.index > spx.index.max())]
    return series


def on_or_before(s, d):
    """Value on the Friday, or the week's last trading day if Friday had no value."""
    window = s.loc[d - timedelta(days=4):d]
    if window.empty:
        raise ValueError(f"no data in the week ending {d.date()}")
    return window.index[-1], float(window.iloc[-1])


def last_friday(today):
    """Most recent Friday whose US close is in; today's Friday counts only from Saturday."""
    return today - timedelta(days=(today.weekday() - 4) % 7 or 7)


def friday_of(day):
    """The Friday on or before a day: the weekly verdict that day belongs to."""
    return day - timedelta(days=(day.weekday() - 4) % 7)


def latest_close(series, now=None):
    """Latest US trading day whose close is in; today counts from 17:00 New York time.

    The calendar is the S&P 500's (VIX's only when the S&P 500 failed to load): Cboe quotes VIX on some stock market
    holidays. Once a week's Friday has passed without a close (a Friday holiday), the answer is that Friday, so the
    page is the weekly one the routine builds for it.
    """
    now = now or datetime.now(ZoneInfo("America/New_York"))
    cutoff = pd.Timestamp(now.date())
    if now.weekday() >= 5 or now.hour < CLOSE_HOUR_ET:
        cutoff -= timedelta(days=1)
    cal = series["spx"].index if series.get("spx") is not None else series["vix"].index
    days = cal[cal <= cutoff]
    if days.empty:
        raise ValueError(f"no closes on or before {cutoff.date()}")
    return max(days[-1], friday_of(cutoff))


def verdict_of(hits):
    return "안정" if hits == 0 else ("경계" if hits <= 2 else "스트레스")


def regime_of(total_bp, hits):
    bp = round(total_bp)
    direction = "상승" if bp >= REGIME_BP else ("하락" if bp <= -REGIME_BP else "횡보")
    name, meaning = REGIMES[(direction, hits > 0)]
    return {"name": name, "meaning": meaning, "dir": direction, "spreading": hits > 0}


# --- plain-language reading: fixed sentence templates filled from the numbers (no free text) ---

def _final(word):
    """Final consonant index of the last Hangul syllable (0 = none)."""
    ch = word[-1]
    return (ord(ch) - 0xAC00) % 28 if "가" <= ch <= "힣" else 0


def _topic(word):
    return word + ("은" if _final(word) else "는")


def _subj(word):
    return word + ("이" if _final(word) else "가")


def _and(words):
    if len(words) == 1:
        return words[0]
    if len(words) == 2:
        return words[0] + ("과" if _final(words[0]) else "와") + " " + words[1]
    return ", ".join(words)


def _bp(v):
    v = round(v)
    return ("+" if v >= 0 else "−") + f"{abs(v)}bp"


def _moved(v):
    return "올랐습니다" if v > 0 else "내렸습니다"


def explain(rec):
    """Two to six plain sentences for a non-expert, built only from rec's numbers.

    Interpretation (is it a rate shock spreading?) is left to the regime line; these sentences only report.
    """
    rate = {r["key"]: r for r in rec["rates"]}
    t, real, be = round(rec["total"]["change_bp"]), round(rate["dfii10"]["change_bp"]), round(rate["t10yie"]["change_bp"])
    out = []
    if abs(t) < 5:
        if abs(real) >= 15 and abs(be) >= 15 and real * be < 0:
            out.append(f"10년 금리는 4주 동안 거의 그대로지만({_bp(t)}), 그 안에서 실질금리({_bp(real)})와 "
                       f"물가 기대(BE {_bp(be)})가 반대로 움직여 서로 상쇄되었습니다.")
        else:
            out.append(f"10년 금리는 4주 동안 거의 그대로입니다({_bp(t)}).")
    else:
        part = "상승분" if t > 0 else "하락분"
        out.append(f"10년 금리가 4주 동안 {abs(t) / 100:.2f}%p {_moved(t)}({_bp(t)}).")
        if real * t > 0 and be * t > 0:
            share = real / (real + be)
            if share >= 0.75:
                out.append(f"{_topic(part)} 대부분 실질금리에서 나왔습니다(실질 {_bp(real)}, BE {_bp(be)}).")
            elif share <= 0.25:
                out.append(f"{_topic(part)} 대부분 물가 기대(BE)에서 나왔습니다(BE {_bp(be)}, 실질 {_bp(real)}).")
            else:
                out.append(f"실질금리({_bp(real)})와 물가 기대(BE {_bp(be)})가 함께 {_moved(t)}.")
        elif real * t > 0:
            tail = "거의 그대로입니다." if abs(be) < 5 else f"오히려 {abs(be)}bp {_moved(be)}."
            out.append(f"{_topic(part)} 모두 실질금리({_bp(real)})에서 나왔고, 물가 기대(BE)는 {tail}")
        elif be * t > 0:
            tail = "거의 그대로입니다." if abs(real) < 5 else f"오히려 {abs(real)}bp {_moved(real)}."
            out.append(f"{_topic(part)} 모두 물가 기대(BE {_bp(be)})에서 나왔고, 실질금리는 {tail}")
        else:  # only on provisional runs, when the 10y and its two parts come from different days
            out.append(f"실질금리({_bp(real)})와 물가 기대(BE {_bp(be)})는 기준일이 달라, 이번에는 10년 금리 변화를 나눠 설명하지 않습니다.")
    desc = {"dgs2": "연준 정책 기대를 보여 주는 2년물", "acmtp10": "국채를 오래 보유하는 대가인 기간프리미엄"}
    size = lambda r: f"{abs(round(r['change_bp']))}bp"  # noqa: E731
    extra = [r for r in sorted(rec["rates"], key=lambda r: r["rank"])[:2]
             if r["key"] in desc and abs(round(r["change_bp"])) >= 10]
    if len(extra) == 1:
        r = extra[0]
        out.append(f"같은 기간 {_topic(desc[r['key']])} {size(r)} {_moved(r['change_bp'])}.")
    elif len(extra) == 2:
        a, b = extra
        if a["change_bp"] * b["change_bp"] > 0:
            out.append(f"같은 기간 {_topic(desc[a['key']])} {size(a)}, {_topic(desc[b['key']])} {size(b)} {_moved(a['change_bp'])}.")
        else:
            out.append(f"같은 기간 {_topic(desc[a['key']])} {size(a)} {_moved(a['change_bp'])[:-3]}고, "
                       f"{_topic(desc[b['key']])} {size(b)} {_moved(b['change_bp'])}.")
    over = [PLAIN[r["key"]] for r in rec["spread"] if r["over"]]
    near = [r for r in rec["spread"] if not r["over"] and r["change_pct"] >= 0.9 * r["threshold"]]
    below = [PLAIN[r["key"]] for r in rec["spread"] if not r["over"] and r not in near]
    near_core = f"{_topic(_and([PLAIN[r['key']] for r in near]))} 기준에 거의 닿았" if near else ""
    if len(near) == 1:
        pct = round(near[0]["change_pct"], 1)
        if pct >= near[0]["threshold"]:  # a near miss must never print as the threshold itself
            pct = math.floor(near[0]["change_pct"] * 10) / 10
        near_num = f"(+{pct:.1f}%, 기준 +{near[0]['threshold']}%)"
    else:
        near_num = ""
    if not over:
        if near:
            out.append(f"위험자산 쪽 네 지표는 모두 기준 아래지만, {near_core}습니다{near_num}.")
        else:
            out.append("위험자산 쪽 네 지표는 모두 기준 아래입니다.")
    elif len(over) == 4:
        out.append("위험자산 쪽 네 지표가 모두 기준을 넘었습니다.")
    else:
        out.append(f"위험자산 쪽에서는 {_subj(_and(over))} 기준을 넘었습니다.")
        if near and below:
            out.append(f"{near_core}고{near_num}, {_topic(_and(below))} 아직 기준 아래입니다.")
        elif near:
            out.append(f"{near_core}습니다{near_num}.")
        else:
            out.append(f"{_topic(_and(below))} 아직 기준 아래입니다.")
    if rec["provisional"] and rec["live"]:
        out.append("일부 지표는 기준일 값이 없어 그 전 거래일 값을 썼습니다. 아직 발표되지 않은 값이라면, 발표된 뒤 다시 갱신할 때 숫자와 판정이 바뀔 수 있습니다.")
    elif rec["provisional"]:
        out.append("일부 지표는 금요일 값이 없어 그 전 거래일 값을 썼으므로, 금요일 값이 나중에 나오면 다음 갱신 때 숫자와 판정이 바뀔 수 있습니다.")
    return out


def note(rec, label, d0, d1, prior, friday):
    """Flag values not taken on the day itself (holiday, or not yet published)."""
    used = [(d, want) for d, want in ((d0, prior), (d1, friday)) if d != want]
    if used and rec["live"]:  # the days compared are not Fridays, so name them
        rec["notes"].append(f"{label}: " + ", ".join(f"{w.month}/{w.day} 값이 없어 {d.month}/{d.day} 값 사용" for d, w in used))
    elif used:
        rec["notes"].append(f"{label}: 금요일 값이 없어 {', '.join(f'{d.month}/{d.day}' for d, _ in used)} 값 사용")
    if d1 != friday:
        rec["provisional"] = True


def compute(series, friday, live=False):
    """The rule for one Friday; with live=True, for any trading day against the same weekday 4 weeks earlier.

    The day is stored under "friday" either way: the page and the records read the as-of day from there.
    """
    prior = friday - timedelta(weeks=4)
    rec = {"friday": str(friday.date()), "prior": str(prior.date()), "rates": [], "spread": [], "notes": [],
           "provisional": False, "live": live}
    key, label, what, _, sid = TOTAL
    d1, v1 = on_or_before(series[key], friday)
    d0, v0 = on_or_before(series[key], prior)
    rec["total"] = {"key": key, "label": label, "what": what, "id": sid, "value": v1, "change_bp": (v1 - v0) * 100,
                    "dates": [str(d0.date()), str(d1.date())]}
    note(rec, label, d0, d1, prior, friday)
    for key, label, what, _, sid in RATES:
        d1, v1 = on_or_before(series[key], friday)
        d0, v0 = on_or_before(series[key], prior)
        rec["rates"].append({"key": key, "label": label, "what": what, "id": sid,
                             "value": v1, "change_bp": (v1 - v0) * 100, "dates": [str(d0.date()), str(d1.date())]})
        note(rec, label, d0, d1, prior, friday)
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
        note(rec, label, d0, d1, prior, friday)
    rec["hits"] = sum(r["over"] for r in rec["spread"])
    rec["verdict"] = verdict_of(rec["hits"])
    rec["cause"] = [ranked[0]["label"], ranked[1]["label"]]
    rec["regime"] = regime_of(rec["total"]["change_bp"], rec["hits"])
    rec["reading"] = explain(rec)
    return rec


def outcome_end(series):
    """Last day both outcome series have data (Saturday runs: HY OAS stops at Thursday)."""
    return min(series["spx"].index.max(), series["hy"].index.max())


def outcome(series, friday):
    """What happened in the SCORE_WEEKS after a Friday: S&P 500 max drawdown (%, <= 0) and HY OAS max widening (bp, >= 0)."""
    end = friday + timedelta(weeks=SCORE_WEEKS)
    if outcome_end(series) < end:
        raise ValueError(f"outcome window after {friday.date()} not complete yet")
    _, p0 = on_or_before(series["spx"], friday)
    _, h0 = on_or_before(series["hy"], friday)
    spx = series["spx"].loc[friday + timedelta(days=1):end]
    hy = series["hy"].loc[friday + timedelta(days=1):end]
    if spx.empty or hy.empty:
        raise ValueError(f"no outcome data after {friday.date()}")
    return {"dd": min(0.0, round((float(spx.min()) / p0 - 1) * 100, 1)),
            "hy": max(0, round((float(hy.max()) - h0) * 100))}


def score_week(series, friday):
    rec = compute(series, friday)
    out = outcome(series, friday)
    return {"d": rec["friday"], "verdict": rec["verdict"], "hits": rec["hits"], "regime": rec["regime"]["name"],
            "dd": out["dd"], "hy": out["hy"], "fell": out["dd"] <= SCORE_DD, "credit": out["hy"] >= SCORE_HY}


def scorecard(series, friday):
    """Every Friday whose 8 following weeks are complete, scored with the current rule.

    Starts where FRED's HY OAS history (a rolling 3 years) allows the full 4-indicator rule.
    """
    base = {"weeks": SCORE_WEEKS, "dd_cut": SCORE_DD, "hy_cut": SCORE_HY}
    if series.get("spx") is None:
        return dict(base, error="spx", summary=[], recent=[], skipped=0, **{"from": None, "to": None})
    first = series["hy"].index.min() + timedelta(weeks=4)
    f = first + timedelta(days=(4 - first.weekday()) % 7)
    last = min(friday, outcome_end(series)) - timedelta(weeks=SCORE_WEEKS)
    rows, skipped = [], 0
    while f <= last:
        try:
            rows.append(score_week(series, f))
        except ValueError:  # a data gap; complete windows only reach here
            skipped += 1
        f += timedelta(weeks=1)
    summary = []
    for v in ["안정", "경계", "스트레스", "전체"]:
        sub = rows if v == "전체" else [r for r in rows if r["verdict"] == v]
        n = len(sub)
        summary.append({"verdict": v, "n": n,
                        "fell": sum(r["fell"] for r in sub) / n * 100 if n else None,
                        "credit": sum(r["credit"] for r in sub) / n * 100 if n else None,
                        "dd_median": float(pd.Series([r["dd"] for r in sub]).median()) if n else None})
    return dict(base, summary=summary, recent=rows[-6:], skipped=skipped,
                **{"from": rows[0]["d"] if rows else None, "to": rows[-1]["d"] if rows else None})


def pad(text, width):
    """Left-align text, counting Korean characters as two columns."""
    cols = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(width - cols, 0)


def render_text(rec):
    lines = [f"날짜: {rec['friday']} (4주 전: {rec['prior']}){' · 주중 잠정' if rec['live'] else ''}", "",
             "[금리]          값        4주 변화"]
    t = rec["total"]
    parts = [r["dates"] for r in rec["rates"] if r["key"] in ("dfii10", "t10yie")]
    same_day = all(p == t["dates"] for p in parts)  # Saturday runs mix Thursday and Friday values
    lines.append(f"{pad(t['label'] + '(합계)', 14)}{t['value']:>6.2f}%   {t['change_bp']:+5.0f}bp   "
                 + ("= 10년 실질 + 10년 BE" if same_day else "≈ 10년 실질 + 10년 BE (기준일이 달라 몇 bp 차이 가능)"))
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
    g = rec["regime"]
    lines.append(f"→ 국면: {g['name']} (10년물 {_bp(t['change_bp'])}, 기준을 넘은 전이 지표 {rec['hits']}개)")
    s = rec.get("score")
    if s and s.get("pending"):
        lines += ["", f"[점검] {SCORE_WEEKS}주 전 {s['d']} 판정: 결과 데이터가 아직 다 나오지 않아 다음 갱신 때 채점합니다"]
    elif s:
        dd = "0.0%" if s["dd"] == 0 else f"{s['dd']:.1f}%"
        lines += ["", f"[점검] {SCORE_WEEKS}주 전 {s['d']} 판정: {s['verdict']} ({s['hits']}/4), 국면: {s['regime']}",
                  f"이후 {SCORE_WEEKS}주 S&P 500 최대 낙폭 {dd}{' ■' if s['fell'] else ''}, "
                  f"HY OAS 최대 확대 {s['hy']}bp{' ■' if s['credit'] else ''} "
                  f"(■ = S&P {abs(SCORE_DD)}% 이상 하락 또는 HY {SCORE_HY}bp 이상 확대)"]
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
    path.write_text(f"# 주간 기록 {rec['friday']}\n\n```\n{render_text(rec)}\n```\n\n"
                    f"자동 해설: {' '.join(rec['reading'])}\n\n메모 한 줄: {memo}\n")
    return path


def weekly_record(series, friday):
    """compute() plus the scorecard line for the verdict made SCORE_WEEKS earlier."""
    rec = compute(series, friday)
    scored = friday - timedelta(weeks=SCORE_WEEKS)
    rec["score"] = None
    if series.get("spx") is not None and scored - timedelta(weeks=4) >= series["hy"].index.min():
        try:
            rec["score"] = score_week(series, scored)
        except ValueError:  # Saturday run (HY OAS stops at Thursday) or a data gap
            rec["score"] = {"pending": True, "d": str(scored.date())}
    return rec


def build_page(series, rec, day):
    """The page for a Friday, or (rec["live"]) for a mid-week close: then the last point of every line and the
    last cell of the verdict strip are that day, and everything weekly (scorecard, memo) stays on its Friday."""
    live = rec["live"]
    friday = friday_of(day) if live else day
    n = SPARK_WEEKS - 1 if live else SPARK_WEEKS  # Fridays per line; a live page adds its day as the last point
    fridays = [friday - timedelta(weeks=k) for k in range(n - 1, -1, -1)]
    points = fridays + [day] if live else fridays
    spark = {}
    for key, *_rest in RATES + SPREAD + [TOTAL]:
        unit_bp = any(row[0] == key and row[5] == "bp" for row in SPREAD)
        spark[key] = [{"d": str(f.date()), "v": on_or_before(series[key], f)[1] * (100 if unit_bp else 1)}
                      for f in points]
    history = []
    for k in range(HISTORY_WEEKS - (2 if live else 1), -1, -1):  # likewise the strip: its last cell is the day
        h = compute(series, friday - timedelta(weeks=k))
        history.append({"d": h["friday"], "hits": h["hits"], "verdict": h["verdict"], "regime": h["regime"]["name"],
                        "over": [r["label"] for r in h["spread"] if r["over"]], "cause": h["cause"], "live": False})
    if live:
        history.append({"d": rec["friday"], "hits": rec["hits"], "verdict": rec["verdict"], "regime": rec["regime"]["name"],
                        "over": [r["label"] for r in rec["spread"] if r["over"]], "cause": rec["cause"], "live": True})
    try:  # the AI credit tab is reference only; its failure (even an import error) must not stop the weekly page
        import aicredit
        ai = aicredit.build(series, day, fred=lambda s: fred(s, tries=1), yahoo=lambda t: yahoo(t, tries=1),
                            points=points, live=live)
    except Exception as e:  # noqa: BLE001 - only the type name: messages can carry request URLs
        ai = {"error": type(e).__name__}
    data = dict(rec, spark=spark, history=history, memo=read_memo(friday), scorecard=scorecard(series, friday), ai=ai,
                regimes=[{"dir": d, "spreading": s, "name": n, "meaning": m} for (d, s), (n, m) in REGIMES.items()],
                regime_bp=REGIME_BP, week_friday=str(friday.date()),
                built_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                refresh={"trigger_id": REFRESH_TRIGGER} if REFRESH_TRIGGER else None)
    template = (ROOT / "page" / "template.html").read_text()
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")  # news text from the web must not end the script
    out = ROOT / "page" / "dashboard.html"
    out.write_text(template.replace("/*__DATA__*/null", payload))
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--latest" in sys.argv:
        if args or "--save" in sys.argv:
            sys.exit("--latest takes no date and is never saved as a weekly record")
        series = load_all()
        day = latest_close(series)
        rec = weekly_record(series, day) if day.weekday() == 4 else compute(series, day, live=True)
        print(render_text(rec))
        if "--page" in sys.argv:
            print(f"built {build_page(series, rec, day)}")
        return
    friday = pd.Timestamp(args[0]) if args else pd.Timestamp(last_friday(date.today()))
    if friday.weekday() != 4:
        sys.exit(f"{friday.date()} is not a Friday")
    series = load_all()
    rec = weekly_record(series, friday)
    print(render_text(rec))
    if "--save" in sys.argv:
        print(f"\nsaved {save_record(rec, friday)}")
        prev = friday - timedelta(weeks=1)
        if record_path(prev).exists():  # replace last week's provisional values with final ones
            print(f"refreshed {save_record(weekly_record(series, prev), prev)}")
    if "--page" in sys.argv:
        print(f"built {build_page(series, rec, friday)}")


if __name__ == "__main__":
    main()
