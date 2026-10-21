"""AI credit watch: the reference tab next to the weekly verdict.

Nothing here feeds the verdict, the regime or the scorecard. Seven views, in the page's order:
  1 (step 1) capex / operating cash flow      SEC filings, cached in data/ai_credit_sec.json
  2 (step 2) distance to default               Yahoo prices + SEC shares and debt
             reported 5-year CDS               data/ai_credit_news.json, written weekly by the routine
             corporate bond prices             FINRA, link only (no free automated access)
  3 (step 3) share price vs the S&P 500        Yahoo
  4 (step 4) BBB corporate spread, BBB - A     FRED
  side       events and alert rule             data/ai_credit_news.json
"""

import json
import math
import os
from datetime import date, timedelta
from pathlib import Path
from statistics import NormalDist

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
SEC_CACHE = ROOT / "data" / "ai_credit_sec.json"
NEWS = ROOT / "data" / "ai_credit_news.json"
COMPANIES = [  # (key, Korean name, ticker, SEC CIK)
    ("orcl", "오라클", "ORCL", "0001341439"),
    ("avgo", "브로드컴", "AVGO", "0001730168"),
    ("nvda", "엔비디아", "NVDA", "0001045810"),
]
BBB, SINGLE_A = "BAMLC0A4CBBB", "BAMLC0A3CA"
SPARK_WEEKS = 13
NEWS_FRESH_DAYS = 14  # an older report is shown as a past report
FINRA_URL = "https://www.finra.org/finra-data/fixed-income"
MATURITY_TAGS = ["LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths",
                 "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearTwo",
                 "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearThree",
                 "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFour",
                 "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFive",
                 "LongTermDebtMaturitiesRepaymentsOfPrincipalAfterYearFive"]


# --- SEC filings (quarterly). SEC requires a User-Agent with a contact email: SEC_USER_AGENT. ---

def _facts(g, tag, duration):
    """10-Q/10-K facts of one tag, the latest filing per period; duration facts have a start date."""
    rows = [x for x in g.get(tag, {}).get("units", {}).get("USD", [])
            if x.get("form") in ("10-Q", "10-K") and ("start" in x) == duration]
    best = {}
    for x in rows:
        k = (x.get("start"), x["end"])
        if k not in best or x.get("filed", "") > best[k].get("filed", ""):
            best[k] = x
    return best


def _latest_instant(g, tag):
    rows = _facts(g, tag, duration=False)
    return max(rows.values(), key=lambda x: x["end"]) if rows else None


def _debt(g):
    """Total long-term debt including current maturities; the newest of three ways companies tag it."""
    found = []
    x = _latest_instant(g, "LongTermDebt")
    if x:
        found.append((x["end"], 2, x["val"], "LongTermDebt"))
    nc, cur = _facts(g, "LongTermDebtNoncurrent", False), _facts(g, "LongTermDebtCurrent", False)
    for k in set(nc) & set(cur):
        found.append((k[1], 1, nc[k]["val"] + cur[k]["val"], "LongTermDebtNoncurrent+Current"))
    parts = [_facts(g, t, False) for t in MATURITY_TAGS]
    for k in set.intersection(*(set(p) for p in parts)) if all(parts) else set():
        found.append((k[1], 0, sum(p[k]["val"] for p in parts), "sum of principal maturities"))
    if not found:
        return None
    end, _, val, how = max(found)
    return {"value": val, "date": end, "how": how}


def _capex_ocf(g):
    ocf = _facts(g, "NetCashProvidedByUsedInOperatingActivities", True)
    for tag in ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"):
        cap = _facts(g, tag, True)
        both = sorted(set(cap) & set(ocf), key=lambda k: (k[1], k[0]))
        if both:
            k = both[-1]
            return {"capex": cap[k]["val"], "ocf": ocf[k]["val"], "start": k[0], "end": k[1], "form": cap[k]["form"]}
    return None


def sec_company(cik, user_agent):
    r = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                     headers={"User-Agent": user_agent}, timeout=90)
    r.raise_for_status()
    facts = r.json()["facts"]
    g = facts["us-gaap"]
    shares = max(facts["dei"]["EntityCommonStockSharesOutstanding"]["units"]["shares"], key=lambda x: x["end"])
    cash = _latest_instant(g, "CashAndCashEquivalentsAtCarryingValue")
    return {"shares": shares["val"], "shares_date": shares["end"], "debt": _debt(g),
            "cash": {"value": cash["val"], "date": cash["end"]} if cash else None, "cashflow": _capex_ocf(g)}


def load_sec():
    """Refresh the cache from SEC when SEC_USER_AGENT is set; otherwise (or on failure) use the cache."""
    cache = json.loads(SEC_CACHE.read_text()) if SEC_CACHE.exists() else {"companies": {}}
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua:
        return cache, f"SEC 자동 갱신이 설정되지 않아 {cache.get('updated', '이전')}에 저장한 공시 값을 씁니다"
    fresh, failed = dict(cache.get("companies", {})), []
    for key, name, _, cik in COMPANIES:
        try:
            fresh[key] = sec_company(cik, ua)
        except Exception as e:  # noqa: BLE001 - keep the cached values for this company
            failed.append(f"{name} ({type(e).__name__})")
    cache = {"updated": str(date.today()), "source": "SEC companyfacts API", "companies": fresh}
    if len(failed) < len(COMPANIES):
        SEC_CACHE.parent.mkdir(exist_ok=True)
        SEC_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1) + "\n")
    return cache, ("SEC에서 받지 못해 저장된 값을 씀: " + ", ".join(failed)) if failed else None


# --- reported CDS and events (written by the weekly routine from news; never computed here) ---

def _https(url):
    return isinstance(url, str) and url.startswith("https://") and len(url) < 500


def load_news(friday):
    if not NEWS.exists():
        return {"as_of": None, "items": [], "events": [], "warning": "보도 자료 파일이 없습니다"}
    try:
        raw = json.loads(NEWS.read_text())
    except ValueError:
        return {"as_of": None, "items": [], "events": [], "warning": "보도 자료 파일을 읽지 못했습니다"}
    keys = {c[0] for c in COMPANIES}
    items, events = [], []
    for x in raw.get("items", []):
        if not (isinstance(x, dict) and x.get("key") in keys and _https(x.get("url")) and isinstance(x.get("source"), str)):
            continue
        bp = x.get("cds_bp")
        if bp is not None and not (isinstance(bp, (int, float)) and 0 < bp < 5000):
            continue
        try:
            age = (friday - pd.Timestamp(x["date"])).days
        except (KeyError, ValueError, TypeError):
            continue
        if age < -4:  # reported after this Friday's Tuesday run: not part of this week's page
            continue
        items.append({"key": x["key"], "cds_bp": bp, "date": str(x["date"])[:10], "record_high": x.get("record_high") is True,
                      "source": x["source"][:120], "url": x["url"], "note": str(x.get("note", ""))[:200],
                      "fresh": age <= NEWS_FRESH_DAYS})
    last = str((friday + timedelta(days=4)).date())
    for e in raw.get("events", []):
        if (isinstance(e, dict) and isinstance(e.get("text"), str) and isinstance(e.get("date"), str)
                and _https(e.get("url")) and e["date"][:10] <= last):
            events.append({"date": e["date"][:10], "text": e["text"][:200], "url": e["url"]})
    events.sort(key=lambda e: e["date"], reverse=True)
    return {"as_of": raw.get("as_of"), "items": items, "events": events[:8], "warning": None}


# --- market data ---

def on_or_before(s, d):
    window = s.loc[d - timedelta(days=4):d]
    if window.empty:
        raise ValueError(f"no data in the week ending {d.date()}")
    return float(window.iloc[-1])


def distance_to_default(prices, shares, debt, rate, d):
    """Naive distance to default (Bharath and Shumway, 2008) over one year; lower means riskier."""
    closes = prices.loc[:d].iloc[-253:]
    sigma_e = closes.apply(math.log).diff().dropna().std() * math.sqrt(252)
    equity = shares * float(closes.iloc[-1])
    sigma_v = equity / (equity + debt) * sigma_e + debt / (equity + debt) * (0.05 + 0.25 * sigma_e)
    return (math.log((equity + debt) / debt) + rate - 0.5 * sigma_v ** 2) / sigma_v


def _won(v):
    return f"{round(v / 1e8):,}억 달러"


def _period(start, end):
    s, e = pd.Timestamp(start) + timedelta(days=7), pd.Timestamp(end) - timedelta(days=7)
    if s.year == e.year:
        return f"{s.year}년 {s.month}월" if s.month == e.month else f"{s.year}년 {s.month}~{e.month}월"
    return f"{s.year}년 {s.month}월~{e.year}년 {e.month}월"


def _final(word):
    ch = word[-1]
    return (ord(ch) - 0xAC00) % 28 if "가" <= ch <= "힣" else 0


def _topic(word):
    return word + ("은" if _final(word) else "는")


def _subj(word):
    return word + ("이" if _final(word) else "가")


def _and(words):
    if len(words) <= 1:
        return "".join(words)
    if len(words) == 2:
        return words[0] + ("과" if _final(words[0]) else "와") + " " + words[1]
    return ", ".join(words)


def build(series, friday, fred, yahoo):
    """Everything the AI tab shows for one Friday. Each block fails on its own and says why."""
    names = {k: n for k, n, _, _ in COMPANIES}
    out = {"friday": str(friday.date()), "companies": [{"key": k, "name": n, "ticker": t} for k, n, t, _ in COMPANIES],
           "finra_url": FINRA_URL, "warnings": []}
    fridays = [friday - timedelta(weeks=k) for k in range(SPARK_WEEKS - 1, -1, -1)]
    prior = friday - timedelta(weeks=4)

    sec, warn = load_sec()
    if warn:
        out["warnings"].append(warn)
    out["sec_updated"] = sec.get("updated")

    # step 1: capex vs operating cash flow
    rows = []
    for key, name, _, _ in COMPANIES:
        c = sec.get("companies", {}).get(key, {})
        cf = c.get("cashflow")
        if cf and cf["ocf"] > 0:
            rows.append({"key": key, "name": name, "capex": cf["capex"], "ocf": cf["ocf"], "ratio": cf["capex"] / cf["ocf"] * 100,
                         "period": _period(cf["start"], cf["end"]), "end": cf["end"],
                         "debt": (c.get("debt") or {}).get("value"), "debt_date": (c.get("debt") or {}).get("date")})
    out["capex"] = rows
    over = [r["name"] for r in rows if r["ratio"] >= 100]
    if not rows:
        out["capex_reading"] = "공시 자료를 받지 못했습니다."
    elif not over:
        out["capex_reading"] = "세 회사 모두 설비투자가 영업현금흐름보다 적습니다."
    else:
        top = max((r for r in rows if r["ratio"] >= 100), key=lambda r: r["ratio"])
        lead = f"{over[0]}만 100%를 넘었습니다." if len(over) == 1 else f"{_subj(_and(over))} 100%를 넘었습니다."
        out["capex_reading"] = (f"{lead} {top['name']}의 {top['period']} 설비투자 {_won(top['capex'])}가 "
                                f"영업현금흐름 {_won(top['ocf'])}보다 많았습니다.")

    # step 3 prices (step 2 needs them too)
    prices, spx = {}, series.get("spx")
    for key, name, ticker, _ in COMPANIES:
        try:
            prices[key] = yahoo(ticker)
        except Exception as e:  # noqa: BLE001
            out["warnings"].append(f"{name} 주가를 받지 못했습니다 ({type(e).__name__})")
    stocks = []
    if spx is not None:
        try:
            spx_chg = (on_or_before(spx, friday) / on_or_before(spx, prior) - 1) * 100
            for key, name, _, _ in COMPANIES:
                if key not in prices:
                    continue
                p = prices[key]
                chg = (on_or_before(p, friday) / on_or_before(p, prior) - 1) * 100
                base = on_or_before(p, fridays[0]) / on_or_before(spx, fridays[0])
                spark = [{"d": str(f.date()), "v": on_or_before(p, f) / on_or_before(spx, f) / base * 100} for f in fridays]
                stocks.append({"key": key, "name": name, "chg": chg, "rel": chg - spx_chg, "spark": spark})
            out["spx_chg"] = spx_chg
        except ValueError as e:
            out["warnings"].append(f"주가 계산 실패 ({e})")
    out["stocks"] = stocks
    if stocks:
        weak = min(stocks, key=lambda r: r["rel"])
        spx_txt = f"{out['spx_chg']:+.1f}%"
        out["stocks_reading"] = (f"같은 4주 동안 S&P 500은 {spx_txt}였고, {_topic(weak['name'])} 시장보다 {abs(weak['rel']):.1f}%p "
                                 "뒤처져 세 회사 가운데 가장 약했습니다." if weak["rel"] < 0 else
                                 f"같은 4주 동안 S&P 500은 {spx_txt}였고, 세 회사 모두 시장보다 앞섰습니다.")
    else:
        out["stocks_reading"] = "주가 자료를 받지 못했습니다."

    # step 2: distance to default, reported CDS
    dd = []
    try:
        rate = on_or_before(series["dgs2"], friday) / 100
        for key, name, _, _ in COMPANIES:
            c = sec.get("companies", {}).get(key, {})
            if key in prices and c.get("shares") and (c.get("debt") or {}).get("value"):
                now = distance_to_default(prices[key], c["shares"], c["debt"]["value"], rate, friday)
                then = distance_to_default(prices[key], c["shares"], c["debt"]["value"], rate, prior)
                dd.append({"key": key, "name": name, "dd": now, "prior": then, "debt_date": c["debt"]["date"]})
    except (ValueError, KeyError) as e:
        out["warnings"].append(f"부도 거리 계산 실패 ({e})")
    out["dd"] = dd
    news = load_news(friday)
    if news["warning"]:
        out["warnings"].append(news["warning"])
    out["news"] = news
    if dd:
        low = min(dd, key=lambda r: r["dd"])
        text = f"세 회사 가운데 부도 거리가 가장 낮은 곳은 {names[low['key']]}입니다({low['dd']:.2f})."
        if round(low["dd"], 2) == round(low["prior"], 2):
            text += " 4주 전과 같습니다."
        else:
            text += f" 4주 전 {low['prior']:.2f}에서 {'낮아졌습니다' if low['dd'] < low['prior'] else '높아졌습니다'}."
        cds = next((x for x in news["items"] if x["key"] == low["key"] and x["cds_bp"] is not None and x["fresh"]), None)
        if cds:
            day = pd.Timestamp(cds["date"])
            text += (f" {day.month}/{day.day}에 보도된 {names[low['key']]} 5년 CDS는 {cds['cds_bp']:.0f}bp"
                     + ("로 사상 최고입니다." if cds["record_high"] else "입니다."))
        out["dd_reading"] = text
    else:
        out["dd_reading"] = "부도 거리를 계산하지 못했습니다."

    # step 4: BBB spread and the BBB - A gap
    try:
        bbb, single_a = fred(BBB), fred(SINGLE_A)
        v1, v0 = on_or_before(bbb, friday), on_or_before(bbb, prior)
        a1, a0 = on_or_before(single_a, friday), on_or_before(single_a, prior)
        out["bbb"] = {"value": v1 * 100, "chg": (v1 - v0) * 100, "gap": (v1 - a1) * 100, "gap_chg": ((v1 - a1) - (v0 - a0)) * 100,
                      "spark": [{"d": str(f.date()), "v": on_or_before(bbb, f) * 100} for f in fridays]}
        chg = round(out["bbb"]["chg"])
        move = "4주 전과 거의 같습니다" if abs(chg) < 1 else f"4주 동안 {abs(chg)}bp {'넓어졌습니다' if chg > 0 else '좁아졌습니다'}"
        out["bbb_reading"] = (f"BBB 스프레드는 {out['bbb']['value']:.0f}bp이고, {move}. "
                              f"BBB와 A 등급의 차이는 {out['bbb']['gap']:.0f}bp입니다.")
    except Exception as e:  # noqa: BLE001 - FRED outage or missing series
        out["bbb"] = None
        out["bbb_reading"] = "이번 갱신에서는 FRED에서 BBB 스프레드를 받지 못했습니다."
        out["warnings"].append(f"BBB 스프레드를 받지 못했습니다 ({type(e).__name__})")
    return out
