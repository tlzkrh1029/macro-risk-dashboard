"""Build a clean calendar-day KRW panel for the risk-budget study.

Inputs
- scratchpad/data/krw_crypto.csv  (Yahoo: KRW=X, BTC-USD, ETH-USD, ^GSPC, SPY adj, ^IRX ...)
- dl/dat89_ko.txt, dl/dat96_ko.txt, dl/dat00_ko.htm (Fed H.10 USD/KRW noon NY, 1981-04+)
- Upbit KRW-BTC / KRW-ETH daily candles (fetched once, cached to dl/upbit_*.csv)

Fixes applied
- Yahoo KRW=X daily bars are stamped at London midnight; during British Summer Time the
  normalised date is one day early (Mon bar labelled Sun ... Fri bar labelled Thu).
  We shift labels inside [last Sun of Mar, last Sun of Oct) by +1 day.
- Data are cut at 2026-10-13 (the 2026-10-14 row is intraday).

Output: panel.pkl (calendar days) with price levels in USD and KRW.
"""
import os, re, time, json
import numpy as np, pandas as pd, requests

HERE = os.path.dirname(os.path.abspath(__file__))
SP = "/tmp/claude-0/-home-user-macro-risk-dashboard/157fe875-3352-5fc5-b8c1-31e705d9a652/scratchpad"
END = pd.Timestamp("2026-10-13")


def last_sunday(y, m):
    d = pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
    return d - pd.Timedelta(days=(d.dayofweek + 1) % 7)


def fix_bst(s):
    idx = []
    for d in s.index:
        a, b = last_sunday(d.year, 3), last_sunday(d.year, 10)
        idx.append(d + pd.Timedelta(days=1) if a <= d < b else d)
    out = pd.Series(s.values, index=pd.DatetimeIndex(idx), name=s.name)
    return out[~out.index.duplicated(keep="last")].sort_index()


def h10():
    rows = []
    for f in ["dat89_ko.txt", "dat96_ko.txt"]:
        for l in open(os.path.join(HERE, "dl", f), encoding="latin-1"):
            m = re.match(r"\s*(\d{1,2}-[A-Za-z]{3}-\d{2})\s+([0-9.]+)\s*$", l)
            if m:
                rows.append((pd.to_datetime(m.group(1), format="%d-%b-%y"), float(m.group(2))))
    t = open(os.path.join(HERE, "dl", "dat00_ko.htm"), encoding="latin-1").read()
    for d, v in re.findall(r"<t[hd][^>]*>\s*([0-9]{1,2}-[A-Z]{3}-[0-9]{2})\s*</t[hd]>\s*<td[^>]*>\s*([^<]*)</td>", t, flags=re.I):
        try:
            rows.append((pd.to_datetime(d, format="%d-%b-%y"), float(v.strip())))
        except ValueError:
            pass
    s = pd.Series(dict(rows)).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.rename("KRW_H10")


def upbit(market):
    path = os.path.join(HERE, "dl", f"upbit_{market}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    UA = {"User-Agent": "Mozilla/5.0"}
    rows, to = [], None
    for _ in range(30):
        url = f"https://api.upbit.com/v1/candles/days?market={market}&count=200" + (f"&to={to}" if to else "")
        j = requests.get(url, headers=UA, timeout=30).json()
        if not j or not isinstance(j, list):
            break
        rows += j
        to = j[-1]["candle_date_time_utc"].replace("T", " ")
        time.sleep(0.2)
    u = pd.DataFrame(rows)
    u["d"] = pd.to_datetime(u.candle_date_time_utc).dt.normalize()
    s = u.drop_duplicates("d").set_index("d").sort_index()["trade_price"].rename(market)
    s.to_csv(path)
    return s


def build():
    raw = pd.read_csv(os.path.join(SP, "data", "krw_crypto.csv"), index_col=0, parse_dates=True)
    raw = raw.loc[:END]
    fx = fix_bst(raw["KRW=X"].dropna())
    cal = pd.date_range("1981-04-13", END, freq="D")
    p = pd.DataFrame(index=cal)
    p["fx_yahoo"] = fx.reindex(cal).ffill(limit=6)
    p["fx_h10"] = h10().reindex(cal).ffill(limit=6)
    # main FX series: Yahoo (close-ish timing) where available, H.10 noon before 2003-12
    p["fx"] = p["fx_yahoo"].combine_first(p["fx_h10"])
    for c, n in [("SPY", "spy"), ("^GSPC", "gspc"), ("BTC-USD", "btc"), ("ETH-USD", "eth"), ("^IRX", "irx")]:
        s = raw[c].dropna()
        p[n] = s.reindex(cal)
        if n in ("spy", "gspc", "irx"):
            p[n] = p[n].ffill(limit=6)  # weekends / holidays: no move
    p["us_open"] = raw["SPY"].dropna().reindex(cal).notna() | raw["^GSPC"].dropna().reindex(cal).notna()
    for m, n in [("KRW-BTC", "btc_upbit"), ("KRW-ETH", "eth_upbit")]:
        try:
            p[n] = upbit(m).reindex(cal)
        except Exception as e:  # noqa
            print("upbit fail", m, e)
    for n in ["spy", "gspc", "btc", "eth"]:
        p[n + "_krw"] = p[n] * p["fx"]
    p.to_pickle(os.path.join(HERE, "panel.pkl"))
    return p


if __name__ == "__main__":
    p = build()
    print(p.dropna(subset=["fx"]).index[0], p.index[-1])
    print(p.loc["2026-10-05":].round(2).to_string())
    # sanity: H10 vs Yahoo agreement after BST fix
    both = p[["fx_yahoo", "fx_h10"]].dropna()
    both = both[p.loc[both.index, "us_open"]]
    r = np.log(both).diff().dropna()
    print("daily log-ret corr Yahoo(BST-fixed) vs H10:", r.corr().iloc[0, 1].round(3),
          " level ratio median", (both.fx_yahoo / both.fx_h10).median().round(4))
    raw = pd.read_csv(os.path.join(SP, "data", "krw_crypto.csv"), index_col=0, parse_dates=True).loc[:END]
    fx0 = raw["KRW=X"].dropna().reindex(both.index)
    r0 = np.log(pd.concat([fx0, both.fx_h10], axis=1)).diff().dropna()
    print("daily log-ret corr Yahoo(raw labels) vs H10:", r0.corr().iloc[0, 1].round(3))
    for n in ["btc", "eth"]:
        k = (p[n + "_upbit"] / (p[n] * p["fx"]) - 1).dropna()
        print(n, "kimchi premium quantiles", k.quantile([.01, .05, .5, .95, .99]).round(3).to_dict(), k.index[0].date())
