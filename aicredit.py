"""AI credit watch: the reference tab next to the weekly verdict.

Nothing here feeds the verdict, the regime or the scorecard, and each block fails on its own. Seven views, in the page's order:
  step 1  capex / operating cash flow            SEC filings, cached in data/ai_credit_sec.json
  step 2  distance to default                    Yahoo prices + SEC shares and debt + 2y Treasury yield
          reported 5-year CDS                    data/ai_credit_news.json, written weekly by the routine from news
          corporate bond prices                  FINRA, link only (no free automated access)
  step 3  share price vs the S&P 500             Yahoo
  step 4  BBB corporate spread, BBB - A          FRED
  side    events and alert rule                  data/ai_credit_news.json
"""

import json
import math
import os
import re
from datetime import date, timedelta
from pathlib import Path

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
CAPEX_TAGS = ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets")
DEBT_PAIRS = [  # (noncurrent, current, rank): balance-sheet pairs, ranked below LongTermDebt and above the maturity sum
    ("LongTermDebtNoncurrent", "LongTermDebtCurrent", 2),
    ("LongTermNotesAndLoans", "NotesPayableCurrent", 1),  # Oracle tags its notes this way every quarter
]
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


def _debt(g):
    """Total long-term debt including current maturities: the newest of the ways companies tag it."""
    found = []
    single = _facts(g, "LongTermDebt", False)
    if single:
        x = max(single.values(), key=lambda x: x["end"])
        found.append((x["end"], 3, x["val"], "LongTermDebt"))
    for noncurrent, current, rank in DEBT_PAIRS:
        a, b = _facts(g, noncurrent, False), _facts(g, current, False)
        for k in set(a) & set(b):
            found.append((k[1], rank, a[k]["val"] + b[k]["val"], f"{noncurrent}+{current}"))
    parts = [_facts(g, t, False) for t in MATURITY_TAGS]
    if all(parts):
        for k in set.intersection(*(set(p) for p in parts)):
            found.append((k[1], 0, sum(p[k]["val"] for p in parts), "sum of principal maturities"))
    if not found:
        return None
    end, _, val, how = max(found)
    return {"value": val, "date": end, "how": how}


def _capex_ocf(g):
    """The newest period that has both capex (under either tag) and operating cash flow."""
    ocf = _facts(g, "NetCashProvidedByUsedInOperatingActivities", True)
    best = None
    for tag in CAPEX_TAGS:
        cap = _facts(g, tag, True)
        for k in set(cap) & set(ocf):
            if best is None or (k[1], k[0]) > (best[0][1], best[0][0]):
                best = (k, cap[k])
    if best is None:
        return None
    k, cap = best
    return {"capex": cap["val"], "ocf": ocf[k]["val"], "start": k[0], "end": k[1], "form": cap["form"]}


def sec_company(cik, user_agent):
    r = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                     headers={"User-Agent": user_agent}, timeout=60)
    r.raise_for_status()
    facts = r.json()["facts"]
    g = facts["us-gaap"]
    shares = max(facts["dei"]["EntityCommonStockSharesOutstanding"]["units"]["shares"], key=lambda x: x["end"])
    cash = _facts(g, "CashAndCashEquivalentsAtCarryingValue", False)
    cash = max(cash.values(), key=lambda x: x["end"]) if cash else None
    return {"shares": shares["val"], "shares_date": shares["end"], "debt": _debt(g),
            "cash": {"value": cash["val"], "date": cash["end"]} if cash else None, "cashflow": _capex_ocf(g)}


def load_sec():
    """Refresh from SEC when SEC_USER_AGENT is set; the cache file changes only when a value changes."""
    try:
        cache = json.loads(SEC_CACHE.read_text()) if SEC_CACHE.exists() else None
    except (OSError, ValueError):
        cache = None
    if not isinstance(cache, dict) or not isinstance(cache.get("companies"), dict):
        cache = None
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua:
        if cache is None:
            return {"companies": {}}, "저장된 SEC 공시 값이 없습니다"
        return cache, f"SEC 자동 갱신이 설정되지 않아 {cache.get('updated', '이전')}에 저장한 공시 값을 씁니다"
    companies, failed, changed = dict((cache or {}).get("companies", {})), [], False
    for key, name, _, cik in COMPANIES:
        try:
            got = sec_company(cik, ua)
        except Exception:  # noqa: BLE001 - keep this company's cached values
            failed.append(name)
            continue
        old = {k: v for k, v in companies.get(key, {}).items() if k != "fetched"}
        if got != old:
            companies[key] = dict(got, fetched=str(date.today()))
            changed = True
    cache = {"updated": str(date.today()) if changed else (cache or {}).get("updated"),
             "source": "SEC companyfacts API", "companies": companies}
    if changed:
        try:
            SEC_CACHE.parent.mkdir(exist_ok=True)
            SEC_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1) + "\n")
        except OSError:
            pass
    return cache, (f"SEC에서 받지 못해 저장된 값을 씁니다: {', '.join(failed)}" if failed else None)


# --- reported CDS and events (written by the weekly routine from news; never computed here) ---

def _https(url):
    return isinstance(url, str) and url.startswith("https://") and len(url) < 500


def _day(s):
    """'YYYY-MM-DD' (a longer ISO timestamp is cut to its date) -> Timestamp, else None."""
    if not isinstance(s, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}([T ].*)?", s):
        return None
    try:
        t = pd.Timestamp(s[:10])
    except ValueError:
        return None
    return None if pd.isna(t) else t


def load_news(friday):
    """Latest valid report per company plus events, for the page of one Friday. Bad entries are skipped."""
    empty = {"as_of": None, "items": [], "events": []}
    if not NEWS.exists():
        return dict(empty, warning="보도 자료 파일이 없습니다")
    try:
        raw = json.loads(NEWS.read_text())
    except (OSError, ValueError):
        return dict(empty, warning="보도 자료 파일을 읽지 못했습니다")
    if not isinstance(raw, dict):
        return dict(empty, warning="보도 자료 파일의 형식이 맞지 않습니다")
    keys = {c[0] for c in COMPANIES}
    latest = {}
    for x in raw.get("items") if isinstance(raw.get("items"), list) else []:
        if not (isinstance(x, dict) and isinstance(x.get("key"), str) and x["key"] in keys):
            continue
        if not (_https(x.get("url")) and isinstance(x.get("source"), str) and x["source"].strip()):
            continue
        bp = x.get("cds_bp")
        if bp is not None and (isinstance(bp, bool) or not isinstance(bp, (int, float)) or not 0 < bp < 5000):
            continue
        day = _day(x.get("date"))
        if day is None or (friday - day).days < -4:  # undated, or reported after this Friday's Tuesday run
            continue
        item = {"key": x["key"], "cds_bp": bp, "date": str(day.date()), "record_high": x.get("record_high") is True,
                "source": x["source"].strip()[:120], "url": x["url"],
                "note": x["note"][:200] if isinstance(x.get("note"), str) else "",
                "fresh": (friday - day).days <= NEWS_FRESH_DAYS}
        if x["key"] not in latest or item["date"] > latest[x["key"]]["date"]:
            latest[x["key"]] = item
    last = str((friday + timedelta(days=4)).date())
    events = []
    for e in raw.get("events") if isinstance(raw.get("events"), list) else []:
        if (isinstance(e, dict) and isinstance(e.get("text"), str) and e["text"].strip() and _https(e.get("url"))
                and isinstance(e.get("date"), str) and re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", e["date"]) and e["date"] <= last):
            events.append({"date": e["date"], "text": e["text"].strip()[:200], "url": e["url"]})
    events.sort(key=lambda e: e["date"], reverse=True)
    as_of = _day(raw.get("as_of"))
    return {"as_of": str(as_of.date()) if as_of is not None else None,
            "items": sorted(latest.values(), key=lambda i: i["date"], reverse=True), "events": events[:8], "warning": None}


# --- market data ---

def on_or_before(s, d):
    window = s.loc[d - timedelta(days=4):d]
    if window.empty:
        raise ValueError(f"no data in the week ending {d.date()}")
    return float(window.iloc[-1])


def distance_to_default(prices, shares, debt, rate, d):
    """Simplified Merton-type distance to default over one year; lower means riskier.

    Bharath and Shumway's (2008) volatility approximation, but with a risk-free drift (the 2y Treasury yield)
    instead of the past-year stock return, and total debt as the default point instead of current + half long-term.
    """
    closes = prices.loc[:d].iloc[-253:]
    sigma_e = closes.apply(math.log).diff().dropna().std() * math.sqrt(252)
    equity = shares * float(closes.iloc[-1])
    sigma_v = equity / (equity + debt) * sigma_e + debt / (equity + debt) * (0.05 + 0.25 * sigma_e)
    return (math.log((equity + debt) / debt) + rate - 0.5 * sigma_v ** 2) / sigma_v


def _won(v):
    return f"{round(v / 1e8):,}억 달러"


def _signed(v):
    return ("+" if v >= 0 else "−") + f"{abs(v):.1f}"


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


def _and(words):
    if len(words) <= 1:
        return "".join(words)
    if len(words) == 2:
        return words[0] + ("과" if _final(words[0]) else "와") + " " + words[1]
    return ", ".join(words)


def _group(names):
    """'세 회사' when every company is covered, else the names that are."""
    return "세 회사" if len(names) == len(COMPANIES) else _and(names)


# --- the tab ---

def _capex_block(sec, out):
    rows, minus, missing = [], [], []
    for key, name, _, _ in COMPANIES:
        c = sec.get("companies", {}).get(key, {})
        cf = c.get("cashflow")
        if not cf:
            missing.append(name)
        elif cf["ocf"] <= 0:
            minus.append(name)
        else:
            rows.append({"key": key, "name": name, "capex": cf["capex"], "ocf": cf["ocf"], "ratio": cf["capex"] / cf["ocf"] * 100,
                         "period": _period(cf["start"], cf["end"]), "end": cf["end"],
                         "debt": (c.get("debt") or {}).get("value"), "debt_date": (c.get("debt") or {}).get("date")})
    if missing:
        out["warnings"].append(f"{_and(missing)} 설비투자·현금흐름 공시 값이 없습니다")
    out["capex"] = rows
    names = [r["name"] for r in rows]
    over = [r for r in rows if r["capex"] > r["ocf"]]
    if not rows:
        text = "" if minus else "공시 자료를 받지 못했습니다."
    elif not over:
        text = (f"{_group(names)} 모두" if len(rows) > 1 else _topic(names[0])) + " 설비투자가 영업현금흐름보다 적습니다."
    else:
        top = max(over, key=lambda r: r["ratio"])
        if len(over) == 1:
            lead = f"{over[0]['name']}만" if len(rows) > 1 else _topic(over[0]["name"])
        else:
            lead = f"{_group([r['name'] for r in over])} 모두"
        text = (f"{lead} 설비투자가 영업현금흐름보다 많았습니다. {'비율이 가장 높은 ' if len(over) > 1 else ''}"
                f"{top['name']}의 {top['period']} 설비투자는 {_won(top['capex'])}로, 영업현금흐름 {_won(top['ocf'])}보다 많았습니다.")
    if minus:
        text = (text + " " if text else "") + f"{_topic(_and(minus))} 영업현금흐름이 0 이하라서 비율을 계산하지 않았습니다."
    out["capex_reading"] = text


def _stocks_block(series, friday, prior, fridays, prices, out):
    stocks, spx = [], series.get("spx")
    try:
        if spx is None:
            raise ValueError("no S&P 500 series")
        spx_chg = (on_or_before(spx, friday) / on_or_before(spx, prior) - 1) * 100
        out["spx_chg"] = spx_chg
    except ValueError:
        out["warnings"].append("S&P 500 자료가 없어 주가를 비교하지 못했습니다")
        out["stocks"], out["stocks_reading"] = [], "주가 자료를 비교하지 못했습니다."
        return
    for key, name, _, _ in COMPANIES:
        if key not in prices:
            continue
        p = prices[key]
        try:
            chg = (on_or_before(p, friday) / on_or_before(p, prior) - 1) * 100
            base = on_or_before(p, fridays[0]) / on_or_before(spx, fridays[0])
            spark = [{"d": str(f.date()), "v": on_or_before(p, f) / on_or_before(spx, f) / base * 100} for f in fridays]
        except ValueError:
            out["warnings"].append(f"{name} 주가에 비어 있는 주가 있어 비교하지 못했습니다")
            continue
        stocks.append({"key": key, "name": name, "chg": chg, "rel": chg - spx_chg, "spark": spark})
    out["stocks"] = stocks
    if not stocks:
        out["stocks_reading"] = "주가 자료를 받지 못했습니다."
        return
    names = [r["name"] for r in stocks]
    weak = min(stocks, key=lambda r: r["rel"])
    lag = [r for r in stocks if r["rel"] <= -0.05]
    head = f"최근 4주 동안 S&P 500은 {_signed(spx_chg)}%였고, "
    if not lag:
        tail = (f"{_group(names)} 모두" if len(stocks) > 1 else _topic(names[0])) + " 시장과 같거나 시장보다 좋았습니다."
    elif len(lag) == len(stocks) > 1:
        tail = f"{_group(names)} 모두 시장보다 뒤처졌고, 가장 많이 뒤처진 곳은 {weak['name']}({abs(weak['rel']):.1f}%p)입니다."
    else:
        tail = f"{_topic(weak['name'])} 시장보다 {abs(weak['rel']):.1f}%p " + ("뒤처져 가장 약했습니다." if len(stocks) > 1 else "뒤처졌습니다.")
    out["stocks_reading"] = head + tail


def _dd_block(series, friday, prior, sec, prices, news, out):
    names = {k: n for k, n, _, _ in COMPANIES}
    dd = []
    try:
        rate, rate_prior = on_or_before(series["dgs2"], friday) / 100, on_or_before(series["dgs2"], prior) / 100
    except (ValueError, KeyError):
        out["warnings"].append("2년물 금리 자료가 없어 부도 거리를 계산하지 못했습니다")
        rate = None
    for key, name, _, _ in COMPANIES:
        if rate is None or key not in prices:
            continue
        c = sec.get("companies", {}).get(key, {})
        debt = (c.get("debt") or {}).get("value")
        if not (c.get("shares") and debt):
            out["warnings"].append(f"{name} 주식 수나 부채 공시 값이 없어 부도 거리를 계산하지 못했습니다")
            continue
        try:
            now = distance_to_default(prices[key], c["shares"], debt, rate, friday)
            then = distance_to_default(prices[key], c["shares"], debt, rate_prior, prior)
        except (ValueError, ZeroDivisionError, IndexError):
            out["warnings"].append(f"{name} 부도 거리를 계산하지 못했습니다")
            continue
        if math.isfinite(now) and math.isfinite(then):
            dd.append({"key": key, "name": name, "dd": now, "prior": then, "debt_date": c["debt"]["date"]})
    out["dd"] = dd
    if not dd:
        out["dd_reading"] = "부도 거리를 계산하지 못했습니다."
        return
    low = min(dd, key=lambda r: r["dd"])
    if len(dd) == 1:
        text = f"부도 거리를 계산한 곳은 {names[low['key']]}뿐이며, 값은 {low['dd']:.2f}입니다."
    else:
        text = f"{_group([r['name'] for r in dd])} 가운데 부도 거리가 가장 낮은 곳은 {names[low['key']]}({low['dd']:.2f})입니다."
    if round(low["dd"], 2) == round(low["prior"], 2):
        text += " 4주 전과 같습니다."
    else:
        text += f" 4주 전 {low['prior']:.2f}에서 {'낮아졌습니다' if low['dd'] < low['prior'] else '높아졌습니다'}."
    cds = next((x for x in news["items"] if x["key"] == low["key"] and x["cds_bp"] is not None and x["fresh"]), None)
    if cds:
        day = pd.Timestamp(cds["date"])
        text += (f" {day.month}월 {day.day}일 보도에 따르면 {names[low['key']]} 5년 CDS는 {cds['cds_bp']:.0f}bp"
                 + ("로, 보도 당시 사상 최고였습니다." if cds["record_high"] else "였습니다."))
    out["dd_reading"] = text


def _bbb_block(friday, prior, fridays, fred, out):
    try:
        bbb, single_a = fred(BBB), fred(SINGLE_A)
        v1, v0 = on_or_before(bbb, friday), on_or_before(bbb, prior)
        a1, a0 = on_or_before(single_a, friday), on_or_before(single_a, prior)
        spark = [{"d": str(f.date()), "v": on_or_before(bbb, f) * 100} for f in fridays]
    except Exception:  # noqa: BLE001 - FRED outage or a missing week; only a Korean note reaches the page
        out["bbb"] = None
        out["bbb_reading"] = "이번 갱신에서는 FRED에서 BBB 스프레드를 받지 못했습니다."
        out["warnings"].append("BBB 스프레드를 받지 못했습니다")
        return
    out["bbb"] = {"value": v1 * 100, "chg": (v1 - v0) * 100, "gap": (v1 - a1) * 100, "gap_chg": ((v1 - a1) - (v0 - a0)) * 100,
                  "spark": spark}
    chg = round(out["bbb"]["chg"])
    move = "4주 전과 거의 같습니다" if abs(chg) < 1 else f"4주 동안 {abs(chg)}bp {'넓어졌습니다' if chg > 0 else '좁아졌습니다'}"
    out["bbb_reading"] = (f"BBB 스프레드는 {out['bbb']['value']:.0f}bp이고, {move}. "
                          f"BBB와 A 등급의 차이는 {out['bbb']['gap']:.0f}bp입니다.")


def build(series, friday, fred, yahoo):
    """Everything the AI tab shows for one Friday. Each block fails on its own and leaves a Korean note."""
    out = {"friday": str(friday.date()), "companies": [{"key": k, "name": n, "ticker": t} for k, n, t, _ in COMPANIES],
           "finra_url": FINRA_URL, "warnings": []}
    fridays = [friday - timedelta(weeks=k) for k in range(SPARK_WEEKS - 1, -1, -1)]
    prior = friday - timedelta(weeks=4)

    sec, warn = load_sec()
    if warn:
        out["warnings"].append(warn)
    out["sec_updated"] = sec.get("updated")
    _capex_block(sec, out)

    prices = {}
    for key, name, ticker, _ in COMPANIES:
        try:
            prices[key] = yahoo(ticker)
        except Exception:  # noqa: BLE001
            out["warnings"].append(f"{name} 주가를 받지 못했습니다")
    try:
        news = load_news(friday)
    except Exception:  # noqa: BLE001 - a news file the routine wrote badly must not take the tab down
        news = {"as_of": None, "items": [], "events": [], "warning": "보도 자료 파일의 형식이 맞지 않습니다"}
    if news["warning"]:
        out["warnings"].append(news["warning"])
    out["news"] = news
    _dd_block(series, friday, prior, sec, prices, news, out)
    _stocks_block(series, friday, prior, fridays, prices, out)
    _bbb_block(friday, prior, fridays, fred, out)
    return out
