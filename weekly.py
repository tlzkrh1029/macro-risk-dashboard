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
REGIMES = {  # (10y direction, spreading?) -> (name, meaning)
    ("상승", True): ("금리발 전이", "금리가 오르면서 그 충격이 위험자산으로 번지고 있습니다."),
    ("횡보", True): ("금리 밖 요인", "금리는 크게 움직이지 않았는데 위험자산이 흔들리고 있습니다."),
    ("하락", True): ("도피형", "금리가 내리는 가운데 위험자산이 흔들리고 있습니다. 불안해진 돈이 국채로 옮겨 가는 모습입니다."),
    ("상승", False): ("질서 있는 금리 상승", "금리는 올랐지만 아직 위험자산으로 번지지 않았습니다."),
    ("횡보", False): ("조용함", "금리도 위험자산도 크게 움직이지 않았습니다."),
    ("하락", False): ("질서 있는 금리 하락", "금리는 내렸고 위험자산도 안정적입니다."),
}


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
           f"?period1={int(time.time()) - YAHOO_DAYS * 86400}&period2={int(time.time())}&interval=1d")
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
    return {row[0]: loaders[row[3]](row[4]) for row in RATES + SPREAD + [TOTAL, OUTCOME]}


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


def _ro(word):
    return word + ("로" if _final(word) in (0, 8) else "으로")


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
    """Two to five plain sentences for a non-expert, built only from rec's numbers."""
    rate = {r["key"]: r for r in rec["rates"]}
    t, real, be = round(rec["total"]["change_bp"]), round(rate["dfii10"]["change_bp"]), round(rate["t10yie"]["change_bp"])
    out = []
    if abs(t) < 5:
        if abs(real) >= 15 and abs(be) >= 15 and real * be < 0:
            out.append(f"10년 금리는 4주 동안 거의 그대로지만({_bp(t)}), 실질금리({_bp(real)})와 물가 기대(BE {_bp(be)})가 "
                       "반대로 움직여 서로 상쇄된 결과입니다.")
        else:
            out.append(f"10년 금리는 4주 동안 거의 그대로입니다({_bp(t)}).")
    else:
        part = "상승분" if t > 0 else "하락분"
        out.append(f"10년 금리가 4주 동안 {abs(t) / 100:.2f}%p {_moved(t)}({_bp(t)}).")
        if real * t > 0 and be * t > 0:
            share = real / t
            if share >= 0.75:
                out.append(f"{_topic(part)} 대부분 실질금리에서 나왔습니다(실질 {_bp(real)}, BE {_bp(be)}).")
            elif share <= 0.25:
                out.append(f"{_topic(part)} 대부분 물가 기대(BE)에서 나왔습니다(BE {_bp(be)}, 실질 {_bp(real)}).")
            else:
                out.append(f"실질금리({_bp(real)})와 물가 기대(BE {_bp(be)})가 함께 {_moved(t)}.")
        elif real * t > 0:
            tail = "거의 그대로입니다." if abs(be) < 5 else f"오히려 {abs(be)}bp {_moved(be)}."
            out.append(f"{_topic(part)} 모두 실질금리에서 나왔고({_bp(real)}), 물가 기대(BE)는 {tail}")
        else:
            tail = "거의 그대로입니다." if abs(real) < 5 else f"오히려 {abs(real)}bp {_moved(real)}."
            out.append(f"{_topic(part)} 모두 물가 기대(BE)에서 나왔고({_bp(be)}), 실질금리는 {tail}")
    desc = {"dgs2": "연준 정책 기대를 보여 주는 2년물", "acmtp10": "국채를 오래 들고 있는 대가인 기간프리미엄"}
    extra = [r for r in sorted(rec["rates"], key=lambda r: r["rank"])[:2]
             if r["key"] in desc and abs(round(r["change_bp"])) >= 10]
    if len(extra) == 1:
        r = extra[0]
        out.append(f"{_topic(desc[r['key']])} {_bp(r['change_bp'])} {_moved(r['change_bp'])}.")
    elif len(extra) == 2:
        a, b = extra
        verb = _moved(a["change_bp"]) if a["change_bp"] * b["change_bp"] > 0 else "움직였습니다"
        out.append(f"{_topic(desc[a['key']])} {_bp(a['change_bp'])}, {_topic(desc[b['key']])} {_bp(b['change_bp'])} {verb}.")
    over = [PLAIN[r["key"]] for r in rec["spread"] if r["over"]]
    near = [r for r in rec["spread"] if not r["over"] and r["change_pct"] >= 0.9 * r["threshold"]]
    below = [PLAIN[r["key"]] for r in rec["spread"] if not r["over"] and r not in near]
    near_core = f"{_topic(_and([PLAIN[r['key']] for r in near]))} 기준에 거의 닿았" if near else ""
    near_num = f"(+{near[0]['change_pct']:.1f}%, 기준 +{near[0]['threshold']}%)" if len(near) == 1 else ""
    if not over:
        if near:
            out.append(f"위험자산 쪽 네 지표는 모두 기준 아래지만, {near_core}습니다{near_num}.")
        else:
            out.append("위험자산 쪽 네 지표가 모두 기준 아래라서, 아직 번지지 않았습니다.")
    elif len(over) == 4:
        out.append("위험자산 쪽 네 경로가 모두 기준을 넘어, 충격이 넓게 번졌습니다.")
    else:
        out.append(f"{_ro(_and(over))}는 번지고 있습니다.")
        if near and below:
            out.append(f"{near_core}고{near_num}, {_topic(_and(below))} 아직 기준 아래입니다.")
        elif near:
            out.append(f"{near_core}습니다{near_num}.")
        else:
            out.append(f"{_topic(_and(below))} 아직 기준 아래입니다.")
    if rec["provisional"]:
        out.append("일부 금리 값은 아직 금요일 확정값이 아니어서, 다음 갱신 때 조금 바뀔 수 있습니다.")
    return out


def note(rec, label, d0, d1, prior, friday):
    """Flag values not taken on the Friday itself (holiday, or not yet published)."""
    used = [f"{d.month}/{d.day}" for d, want in ((d0, prior), (d1, friday)) if d != want]
    if used:
        rec["notes"].append(f"{label}: 금요일 값이 없어 {', '.join(used)} 값 사용")
        if d1 != friday:
            rec["provisional"] = True


def compute(series, friday):
    prior = friday - timedelta(weeks=4)
    rec = {"friday": str(friday.date()), "prior": str(prior.date()), "rates": [], "spread": [], "notes": [],
           "provisional": False}
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


def outcome(series, friday):
    """What happened in the SCORE_WEEKS after a Friday: S&P 500 max drawdown (%) and HY OAS max widening (bp)."""
    end = friday + timedelta(weeks=SCORE_WEEKS)
    _, p0 = on_or_before(series["spx"], friday)
    _, h0 = on_or_before(series["hy"], friday)
    spx = series["spx"].loc[friday + timedelta(days=1):end]
    hy = series["hy"].loc[friday + timedelta(days=1):end]
    if spx.empty or hy.empty:
        raise ValueError(f"no outcome data after {friday.date()}")
    return {"dd": (float(spx.min()) / p0 - 1) * 100, "hy": (float(hy.max()) - h0) * 100}


def score_week(series, friday):
    rec = compute(series, friday)
    out = outcome(series, friday)
    return {"d": rec["friday"], "verdict": rec["verdict"], "hits": rec["hits"], "regime": rec["regime"]["name"],
            "dd": out["dd"], "hy": out["hy"], "fell": out["dd"] <= SCORE_DD, "credit": out["hy"] >= SCORE_HY}


def scorecard(series, friday):
    """Every Friday whose 8 following weeks are complete, scored with the same rule.

    Starts where FRED's HY OAS history (a rolling 3 years) allows the full 4-indicator rule.
    """
    first = series["hy"].index.min() + timedelta(weeks=4)
    f = first + timedelta(days=(4 - first.weekday()) % 7)
    rows, skipped = [], 0
    while f <= friday - timedelta(weeks=SCORE_WEEKS):
        try:
            rows.append(score_week(series, f))
        except ValueError:
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
    return {"from": rows[0]["d"] if rows else None, "to": rows[-1]["d"] if rows else None, "skipped": skipped,
            "weeks": SCORE_WEEKS, "dd_cut": SCORE_DD, "hy_cut": SCORE_HY, "summary": summary, "recent": rows[-6:]}


def pad(text, width):
    """Left-align text, counting Korean characters as two columns."""
    cols = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(width - cols, 0)


def render_text(rec):
    lines = [f"날짜: {rec['friday']} (4주 전: {rec['prior']})", "", "[금리]          값        4주 변화"]
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
    if s:
        lines += ["", f"[점검] {SCORE_WEEKS}주 전 {s['d']} 판정: {s['verdict']} ({s['hits']}/4), 국면: {s['regime']}",
                  f"이후 {SCORE_WEEKS}주 S&P 500 최대 낙폭 {s['dd']:+.1f}%{' ■' if s['fell'] else ''}, "
                  f"HY OAS 최대 확대 {_bp(s['hy'])}{' ■' if s['credit'] else ''} "
                  f"(■ = S&P {SCORE_DD}% 이하 또는 HY +{SCORE_HY}bp 이상)"]
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
    try:
        rec["score"] = score_week(series, friday - timedelta(weeks=SCORE_WEEKS))
    except ValueError:
        rec["score"] = None
    return rec


def build_page(series, rec, friday):
    fridays = [friday - timedelta(weeks=k) for k in range(SPARK_WEEKS - 1, -1, -1)]
    spark = {}
    for key, *_rest in RATES + SPREAD + [TOTAL]:
        unit_bp = any(row[0] == key and row[5] == "bp" for row in SPREAD)
        spark[key] = [{"d": str(f.date()), "v": on_or_before(series[key], f)[1] * (100 if unit_bp else 1)}
                      for f in fridays]
    history = []
    for k in range(HISTORY_WEEKS - 1, -1, -1):
        f = friday - timedelta(weeks=k)
        h = compute(series, f)
        history.append({"d": h["friday"], "hits": h["hits"], "verdict": h["verdict"], "regime": h["regime"]["name"],
                        "over": [r["label"] for r in h["spread"] if r["over"]], "cause": h["cause"]})
    data = dict(rec, spark=spark, history=history, memo=read_memo(friday), scorecard=scorecard(series, friday),
                regimes=[{"dir": d, "spreading": s, "name": n, "meaning": m} for (d, s), (n, m) in REGIMES.items()],
                regime_bp=REGIME_BP)
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
