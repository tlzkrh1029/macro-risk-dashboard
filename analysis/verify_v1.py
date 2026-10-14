"""Reproduce the quantitative checks behind docs/review-2026-10-14.md.

Data sources (all free, no API key):
  - Federal Reserve GSW nominal and TIPS zero-coupon curves (stand-ins for DGS10/DFII10/T10YIE)
  - NY Fed ACM term premium (daily)
  - Yahoo Finance daily closes (MOVE, VIX, ETFs, DXY, BTC, equities)

FRED is not used because it was unreachable from the review environment.
Note: the GSW curves are zero-coupon yields, so levels differ by a few bp
from FRED's constant-maturity par yields; changes are nearly identical.

Usage:
    pip install pandas numpy scipy xlrd requests
    python analysis/verify_v1.py            # downloads into analysis/.cache/
    python analysis/verify_v1.py --offline  # reuse cached files
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import percentileofscore

CACHE = Path(__file__).resolve().parent / ".cache"
UA = {"User-Agent": "Mozilla/5.0"}
DOC_DATE = "2026-08-19"  # date of docs/v1-design.md

FED_NOMINAL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
FED_TIPS = "https://www.federalreserve.gov/data/yield-curve-tables/feds200805.csv"
ACM_XLS = "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xls"
TICKERS = [
    "^MOVE", "^TNX", "^TYX", "^VIX", "^VIX3M", "HYG", "LQD", "IEI", "IEF", "TLT",
    "VCLT", "VGLT", "HYGH", "DX-Y.NYB", "BTC-USD", "GLD", "^GSPC", "^IXIC",
]


def download(offline):
    CACHE.mkdir(exist_ok=True)
    if not offline:
        for url, name in [(FED_NOMINAL, "gsw_nominal.csv"), (FED_TIPS, "gsw_tips.csv"), (ACM_XLS, "acm.xls")]:
            r = requests.get(url, headers=UA, timeout=180)
            r.raise_for_status()
            (CACHE / name).write_bytes(r.content)
        closes = {}
        for t in TICKERS:
            url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{requests.utils.quote(t)}"
                   f"?period1=0&period2={int(time.time())}&interval=1d&includeAdjustedClose=true")
            j = requests.get(url, headers=UA, timeout=60).json()["chart"]["result"][0]
            idx = pd.to_datetime(j["timestamp"], unit="s").normalize()
            px = j["indicators"].get("adjclose", [{}])[0].get("adjclose") or j["indicators"]["quote"][0]["close"]
            s = pd.Series(px, index=idx, name=t).dropna()
            closes[t] = s[~s.index.duplicated(keep="last")]
        pd.DataFrame(closes).to_csv(CACHE / "yahoo.csv")


def read_gsw(path):
    with open(path) as f:
        skip = next(i for i, line in enumerate(f) if line.startswith("Date,"))
    return pd.read_csv(path, skiprows=skip, index_col=0, parse_dates=True).apply(pd.to_numeric, errors="coerce")


def load():
    nom = read_gsw(CACHE / "gsw_nominal.csv")
    tips = read_gsw(CACHE / "gsw_tips.csv")
    acm = pd.read_excel(CACHE / "acm.xls", sheet_name="ACM Daily")
    acm["DATE"] = pd.to_datetime(acm["DATE"], format="%d-%b-%Y")
    acm = acm.set_index("DATE")
    y = pd.read_csv(CACHE / "yahoo.csv", index_col=0, parse_dates=True)
    # BTC trades on weekends; keep weekdays only so daily diffs line up across assets.
    y = y[y.index.dayofweek < 5].ffill(limit=3)
    return nom, tips, acm, y


def at(s, date):
    return s.dropna().loc[:date].iloc[-1]


def section(title):
    print(f"\n=== {title} ===")


def decomposition(nom, tips, acm):
    section("1. 10y yield decomposition (bp changes)")
    c = pd.DataFrame({
        "nominal": nom["SVENY10"], "real": tips["TIPSY10"], "breakeven": tips["BKEVEN10"],
        "acm_term_premium": acm["ACMTP10"], "acm_expected_path": acm["ACMRNY10"], "acm_2y": acm["ACMY02"],
    }).dropna()
    last = c.index[-1].strftime("%Y-%m-%d")
    for a, b in [("2026-04-30", DOC_DATE), (DOC_DATE, last), ("2026-09-11", last)]:
        chg = ((at(c, b) - at(c, a)) * 100).round(0).astype(int)
        print(f"{a} -> {b}: " + ", ".join(f"{k} {v:+d}" for k, v in chg.items()))
    w = c.resample("W-FRI").last()
    q = (w.diff(4) * 100).loc["2004":].abs().quantile([0.5, 0.8, 0.9]).round(0)
    print("abs 4-week change percentiles since 2004 (bp):")
    print(q.to_string())


def percentiles(tips, acm, y):
    section("2. Level percentiles (3y / 10y / full sample)")
    series = {"30y (^TYX)": y["^TYX"], "MOVE": y["^MOVE"], "VIX": y["^VIX"],
              "10y real (GSW)": tips["TIPSY10"], "ACM TP10": acm["ACMTP10"]}
    for name, s in series.items():
        s = s.dropna()
        for d in [DOC_DATE, s.index[-1]]:
            hist = s.loc[:d]
            v = hist.iloc[-1]
            out = [percentileofscore(hist.loc[pd.Timestamp(d) - pd.DateOffset(years=n):], v, kind="weak")
                   for n in (3, 10)] + [percentileofscore(hist, v, kind="weak")]
            print(f"{name:15s} {pd.Timestamp(d).date()} value {v:7.2f}  pct " + " / ".join(f"{p:5.1f}" for p in out))
    for name, s in [("30y (^TYX)", y["^TYX"]), ("10y real (GSW)", tips["TIPSY10"]), ("ACM TP10", acm["ACMTP10"])]:
        s = s.dropna()
        for d in [DOC_DATE, s.index[-1]]:
            v = at(s, d)
            before = s.loc[:"2025-12-31"]
            print(f"{name} {pd.Timestamp(d).date()} = {v:.2f}; last time at or above before 2026: "
                  f"{before[before >= v].index[-1].date()}")
    m = y["^MOVE"].dropna()
    print(f"MOVE mean {m.mean():.1f}, median {m.median():.1f} (since {m.index[0].date()})")


def move_history(y):
    section("3. MOVE in the v1 case-table examples")
    m = y["^MOVE"].dropna()
    for a, b, label in [("2019-05-01", "2019-08-31", "mid-2019 (case A)"), ("2020-04-01", "2020-04-30", "2020-04 (case B)"),
                        ("2022-10-01", "2022-10-31", "2022-10 (case B)"), ("2022-01-01", "2022-06-30", "2022 H1 (case D)")]:
        s = m.loc[a:b]
        print(f"{label:18s} min {s.min():6.1f} mean {s.mean():6.1f} max {s.max():6.1f}")
    d = pd.DataFrame({"move": m, "spx": y["^GSPC"]}).dropna()
    d["dd"] = d.spx / d.spx.rolling(252, min_periods=1).max() - 1
    d["fwd63"] = d.spx.shift(-63) / d.spx - 1
    hi = d[d.move > 126]
    print(f"MOVE>126 days: {len(hi)}; share with SPX >10% below 52w high: {(hi.dd < -0.10).mean():.0%} "
          f"(all days {(d.dd < -0.10).mean():.0%}); fwd 63d SPX {hi.fwd63.mean():+.1%} vs all {d.fwd63.mean():+.1%}")
    r5 = m.pct_change(5).loc["2026-09":]
    fri = m.resample("W-FRI").last().pct_change().loc["2026-09":]
    print(f"Sep-Oct 2026 MOVE: max rolling 5d change {r5.max():+.0%} ({r5.idxmax().date()}), max Fri-to-Fri {fri.max():+.0%}")


def proxy_contamination(y):
    section("4. Rate contamination of credit ETF ratios (weekly, 2015+)")
    lg = np.log(y[["HYG", "LQD", "IEI", "IEF", "TLT", "VCLT", "VGLT", "HYGH"]]).diff()
    ratios = pd.DataFrame({
        "HYG/LQD (v1)": lg.HYG - lg.LQD, "HYG/IEI": lg.HYG - lg.IEI, "HYGH": lg.HYGH,
        "VCLT/TLT (v1)": lg.VCLT - lg.TLT, "VCLT/VGLT": lg.VCLT - lg.VGLT, "LQD/IEF": lg.LQD - lg.IEF,
        "d10y": y["^TNX"].diff(),
    }).loc["2015":].dropna().resample("W-FRI").sum()
    for k in ratios.columns.drop("d10y"):
        beta = np.polyfit(ratios.d10y, ratios[k] * 100, 1)[0]
        r2 = np.corrcoef(ratios.d10y, ratios[k])[0, 1] ** 2
        print(f"{k:14s} {beta:+.2f}% per +100bp in 10y, R2 {r2:.2f}")

    def lr(a, b, k):
        return np.log(at(y[k], b) / at(y[k], a)) * 100

    a, b = "2026-08-28", "2026-10-01"  # HY OAS ~260bp -> ~324bp per secondary reports of FRED data
    print(f"{a} -> {b}: HYG/LQD {lr(a, b, 'HYG') - lr(a, b, 'LQD'):+.2f}%, VCLT/TLT {lr(a, b, 'VCLT') - lr(a, b, 'TLT'):+.2f}%, "
          f"HYG/IEI {lr(a, b, 'HYG') - lr(a, b, 'IEI'):+.2f}%, HYGH {lr(a, b, 'HYGH'):+.2f}%, LQD/IEF {lr(a, b, 'LQD') - lr(a, b, 'IEF'):+.2f}%")


def crypto(tips, acm, y):
    section("5. BTC weekly correlations")
    w = pd.DataFrame({
        "btc": np.log(y["BTC-USD"]), "ndx": np.log(y["^IXIC"]), "dxy": np.log(y["DX-Y.NYB"]),
        "gld": np.log(y["GLD"]), "real": tips["TIPSY10"], "be": tips["BKEVEN10"], "tp": acm["ACMTP10"],
    }).ffill().resample("W-FRI").last().diff().loc["2016":].dropna()
    rows = []
    for yr in range(2016, w.index[-1].year + 1):
        s = w.loc[str(yr)]
        rows.append([yr] + [round(s.btc.corr(s[k]), 2) for k in ["real", "be", "tp", "dxy", "ndx", "gld"]])
    print(pd.DataFrame(rows, columns=["year", "dReal", "dBE", "dTP", "dDXY", "NDX", "GLD"]).to_string(index=False))
    s = w.loc["2018":]
    print("R2 2018+: " + ", ".join(f"{k} {np.corrcoef(s.btc, s[k])[0, 1] ** 2:.3f}" for k in ["real", "be", "ndx", "dxy"]))
    rc = w.btc.rolling(52).corr(w.real).loc["2017":]
    print(f"rolling 52w corr(BTC, dReal): median {rc.median():+.2f}, last {rc.iloc[-1]:+.2f}; "
          f"corr(BTC, NDX) last {w.btc.rolling(52).corr(w.ndx).iloc[-1]:+.2f}")
    for k in ["BTC-USD", "^GSPC", "DX-Y.NYB"]:
        print(f"{k} {DOC_DATE} -> last: {at(y[k], y[k].dropna().index[-1]) / at(y[k], DOC_DATE) - 1:+.1%}")


def hedge(y):
    section("6. Stock-bond hedge (63d corr of SPX and TLT daily returns; >0 = bonds not hedging)")
    d = np.log(y[["^GSPC", "TLT"]]).diff().dropna()
    sb = d["^GSPC"].rolling(63).corr(d["TLT"])
    print(f"last {sb.iloc[-1]:+.2f}; " + ", ".join(f"{yr} mean {sb.loc[str(yr)].mean():+.2f}" for yr in [2019, 2022, 2024, 2025, 2026]))


def composite_backtest(y):
    section("7. Composite alarm backtest (weekly, 2008+; credit proxy = HYG/IEI)")
    w = y[["HYG", "IEI", "^MOVE", "TLT", "DX-Y.NYB", "^GSPC", "BTC-USD", "^VIX"]].ffill().resample("W-FRI").last().loc["2008":]
    L = np.log(w)
    credit = (L.HYG - L.IEI).diff()
    dm, dt, dd, dv = L["^MOVE"].diff(), L.TLT.diff(), L["DX-Y.NYB"].diff(), L["^VIX"].diff()
    spx = w["^GSPC"].values
    mdd13 = pd.Series([spx[i:i + 14].min() / spx[i] - 1 if i + 1 < len(spx) else np.nan for i in range(len(spx))], index=w.index)
    fwd = pd.DataFrame({
        "spx4": L["^GSPC"].shift(-4) - L["^GSPC"], "spx13": L["^GSPC"].shift(-13) - L["^GSPC"], "mdd13": mdd13,
        "btc4": L["BTC-USD"].shift(-4) - L["BTC-USD"],
    })
    signals = {
        "unconditional": pd.Series(True, index=w.index),
        "v1 composite, direction only": (credit < 0) & (dm > 0) & (dt < 0) & (dd > 0),
        "A: rate-led, with magnitudes": (credit < -0.005) & (dm > 0.10) & (dt < -0.01) & (dd > 0.005),
        "B: flight-to-quality": (credit < -0.005) & (dm > 0.10) & (dt > 0.01) & (dv > 0.15),
    }
    rows = []
    for name, s in signals.items():
        s = s.fillna(False)
        f = fwd[s]
        rows.append([name, int(s.sum()), f"{s.mean():.1%}", f"{f.spx4.mean():+.2%}", f"{f.spx13.mean():+.2%}",
                     f"{f.mdd13.median():+.1%}", f"{(f.mdd13 < -0.10).mean():.0%}", f"{f.btc4.loc['2015':].mean():+.1%}"])
    print(pd.DataFrame(rows, columns=["signal", "weeks", "share", "SPX 4w", "SPX 13w", "med MDD 13w",
                                      "P(MDD<-10%)", "BTC 4w"]).to_string(index=False))
    for name in ["A: rate-led, with magnitudes", "B: flight-to-quality"]:
        s = signals[name].fillna(False)
        print(f"{name} fired: {', '.join(str(d.date()) for d in s[s].index)}")
    s = signals["v1 composite, direction only"].fillna(False).loc["2026"]
    print(f"v1 direction-only fired in 2026: {', '.join(str(d.date()) for d in s[s].index)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--offline", action="store_true")
    args = p.parse_args()
    download(args.offline)
    nom, tips, acm, y = load()
    print(f"data through: GSW {tips.dropna(how='all').index[-1].date()}, ACM {acm.index[-1].date()}, Yahoo {y.index[-1].date()}")
    decomposition(nom, tips, acm)
    percentiles(tips, acm, y)
    move_history(y)
    proxy_contamination(y)
    crypto(tips, acm, y)
    hedge(y)
    composite_backtest(y)


if __name__ == "__main__":
    sys.exit(main())
