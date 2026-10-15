"""Section 1: KRW-terms monthly tails, mixes, and the USD/KRW cushion."""
import numpy as np, pandas as pd
from lib import panel, tails, fmt_pct

p = panel()
END_M = "2026-09"
M = p.drop(columns=["us_open"]).astype(float).resample("ME").last()
mr = M.pct_change(fill_method=None)
pd.set_option("display.width", 250)


def row(name, s, nav=None):
    t = tails(s.dropna(), nav)
    return {"series": name, "n": t["n"], "p5": fmt_pct(t["p5"]), "p1": fmt_pct(t["p1"]), "worst": f"{fmt_pct(t['worst_m'])} ({t['worst_m_date']})",
            "worst3m": f"{fmt_pct(t['worst_3m'])} (to {t['worst_3m_end']})", "vol": fmt_pct(t["vol"], 0),
            "k99=|p1|/sd_m": f"{t['k99']:.2f}", "k95": f"{t['k95']:.2f}", "maxDD(daily)": fmt_pct(t.get("maxdd", np.nan))}


def dnav(col, a, b):
    s = p[col].loc[a:b].dropna()
    return s / s.iloc[0]


rows = []
E0, E1 = "2017-12", END_M
for nm, cu, ck in [("SPY", "spy", "spy_krw"), ("BTC", "btc", "btc_krw"), ("ETH", "eth", "eth_krw")]:
    rows.append(row(f"{nm} USD 2017-12..", mr[cu].loc[E0:E1], dnav(cu, "2017-11-30", "2026-09-30")))
    rows.append(row(f"{nm} KRW 2017-12..", mr[ck].loc[E0:E1], dnav(ck, "2017-11-30", "2026-09-30")))
for nm, ck in [("BTC Upbit KRW", "btc_upbit"), ("ETH Upbit KRW", "eth_upbit")]:
    rows.append(row(f"{nm} 2017-12..", mr[ck].loc[E0:E1], dnav(ck, "2017-11-30", "2026-09-30")))
for a, lab in [("2004-01", "2004+"), ("1993-02", "1993+ (H.10 FX pre-2004)")]:
    rows.append(row(f"SPY USD {lab}", mr["spy"].loc[a:E1], dnav("spy", a, "2026-09-30")))
    rows.append(row(f"SPY KRW {lab}", mr["spy_krw"].loc[a:E1], dnav("spy_krw", a, "2026-09-30")))
import os
_raw = pd.read_csv("/tmp/claude-0/-home-user-macro-risk-dashboard/157fe875-3352-5fc5-b8c1-31e705d9a652/scratchpad/data/krw_crypto.csv", index_col=0, parse_dates=True)["^GSPC"].dropna().loc[:"2026-09-30"]
_g = _raw.resample("ME").last().pct_change(fill_method=None).dropna()
rows.append(row("^GSPC USD (price) 1970+", _g.loc["1970-02":E1], _raw / _raw.iloc[0]))
rows.append(row("^GSPC USD (price) 1981-05+", mr["gspc"].loc["1981-05":E1]))
rows.append(row("^GSPC KRW (price) 1981-05+", mr["gspc_krw"].loc["1981-05":E1], dnav("gspc_krw", "1981-04-30", "2026-09-30")))
T1 = pd.DataFrame(rows)
print("=== T1 single-asset monthly tails ===")
print(T1.to_string(index=False))

# ---------------- mixes (monthly rebalanced) ----------------
r = mr.loc["2017-12":E1]
# risk-parity BTC:ETH weights from trailing 365d daily vol at prior month end (past data only)
dv = p[["btc_krw", "eth_krw"]].pct_change(fill_method=None)
rv = dv.rolling(365, min_periods=60).std().resample("ME").last().shift(1)
w_btc_rp = (1 / rv["btc_krw"]) / (1 / rv["btc_krw"] + 1 / rv["eth_krw"])
w_btc_rp = w_btc_rp.reindex(r.index)
rows = []
mixes = {}
for ccy, sfx in [("USD", ""), ("KRW", "_krw")]:
    for eq in [1.0, 0.9, 0.85, 0.8]:
        for split in ["2:1", "RP"]:
            if eq == 1.0 and split == "RP":
                continue
            wb = (2 / 3 if split == "2:1" else w_btc_rp)
            cr = wb * r["btc" + sfx] + (1 - wb) * r["eth" + sfx]
            m = eq * r["spy" + sfx] + (1 - eq) * cr
            name = f"{int(eq*100)}/{int(round((1-eq)*100))} {split} {ccy}"
            mixes[name] = m
            nav = (1 + m).cumprod()
            rows.append(row(name, m, nav))
T2 = pd.DataFrame(rows)
print("\n=== T2 mixes, monthly rebalanced, 2017-12..2026-09 (maxDD here is on month-end NAV) ===")
print(T2.to_string(index=False))
print("avg BTC weight in risk-parity crypto sleeve:", round(w_btc_rp.mean(), 3))

# Upbit vs USDxFX for the crypto sleeve
cr_up = 2 / 3 * r["btc_upbit"] + 1 / 3 * r["eth_upbit"]
cr_fx = 2 / 3 * r["btc_krw"] + 1 / 3 * r["eth_krw"]
m_up = 0.85 * r["spy_krw"] + 0.15 * cr_up
print("\n85/15 2:1 KRW with Upbit crypto prices:", row("85/15 Upbit", m_up, (1 + m_up).cumprod()))
d = (cr_up - cr_fx)
print("crypto sleeve Upbit minus USDxFX monthly diff: worst", fmt_pct(d.min()), d.idxmin().strftime("%Y-%m"),
      " best", fmt_pct(d.max()), d.idxmax().strftime("%Y-%m"), " sd", fmt_pct(d.std()))
print(d.sort_values().head(5).apply(fmt_pct).to_string())

# ---------------- cushion ----------------
print("\n=== T3 cushion in SPY stress months (USD return <= -5%) ===")
fxm = mr["fx"]
out = []
for a, b, lab in [("1993-02", "2003-12", "1993-2003 (H.10)"), ("2004-01", "2009-12", "2004-09"), ("2010-01", "2016-12", "2010-16"),
                  ("2017-01", "2021-12", "2017-21"), ("2022-01", E1, "2022-26"), ("1993-02", E1, "all 1993+")]:
    s = mr.loc[a:b]
    st = s[s.spy <= -0.05]
    allc = np.corrcoef(s.spy.dropna(), s.fx.reindex(s.spy.dropna().index))[0, 1]
    beta = np.polyfit(s.spy.dropna(), s.fx.reindex(s.spy.dropna().index), 1)[0]
    out.append({"period": lab, "n_stress": len(st), "avg SPY USD": fmt_pct(st.spy.mean()), "avg FX (KRW/USD)": fmt_pct(st.fx.mean()),
                "avg SPY KRW": fmt_pct(st.spy_krw.mean()), "cushion %pts": fmt_pct(st.spy_krw.mean() - st.spy.mean()),
                "share of loss offset": fmt_pct(1 - st.spy_krw.mean() / st.spy.mean(), 0) if len(st) else "",
                "monthly corr(SPY,FX)": f"{allc:.2f}", "beta FX on SPY": f"{beta:.2f}"})
print(pd.DataFrame(out).to_string(index=False))

print("\n=== T3b cushion in BTC stress months (BTC USD <= -20%), 2017-12.. ===")
s = mr.loc["2014-10":E1]
for a, b, lab in [("2014-10", "2019-12", "2014-19"), ("2020-01", E1, "2020-26"), ("2014-10", E1, "all")]:
    x = s.loc[a:b]
    st = x[x.btc <= -0.2]
    c = x[["btc", "fx"]].dropna().corr().iloc[0, 1]
    print(f"{lab}: n={len(st)} avg BTC USD {fmt_pct(st.btc.mean())} FX {fmt_pct(st.fx.mean())} BTC KRW {fmt_pct(st.btc_krw.mean())}  monthly corr(BTC,FX) {c:.2f}")

# rolling correlations (weekly returns, 156w) and (monthly, 36m)
wk = p[["spy", "btc", "fx", "eth"]].resample("W-FRI").last().pct_change(fill_method=None)
rc_spy = wk.spy.rolling(156, min_periods=100).corr(wk.fx)
rc_btc = wk.btc.rolling(156, min_periods=100).corr(wk.fx)
mc_spy = mr.spy.rolling(36).corr(mr.fx)
mc_btc = mr.btc.rolling(36).corr(mr.fx)
print("\n=== T4 rolling correlation with USD/KRW (3y weekly, 3y monthly), year-end values ===")
ye = pd.DataFrame({"SPY 156w": rc_spy, "BTC 156w": rc_btc}).resample("YE").last()
ym = pd.DataFrame({"SPY 36m": mc_spy, "BTC 36m": mc_btc}).resample("YE").last()
print(pd.concat([ye, ym], axis=1).loc["1996":].round(2).to_string())
print("latest (2026-10-09 wk / 2026-09 m):", round(rc_spy.loc[:"2026-10-09"].iloc[-1], 2), round(rc_btc.loc[:"2026-10-09"].iloc[-1], 2),
      round(mc_spy.loc[:"2026-09"].iloc[-1], 2), round(mc_btc.loc[:"2026-09"].iloc[-1], 2))
# weekly beta by window
for a, b in [("2004", "2009"), ("2010", "2016"), ("2017", "2021"), ("2022", "2026"), ("2024", "2026")]:
    x = wk.loc[a:b].dropna(subset=["spy", "fx"])
    bt = np.polyfit(x.spy, x.fx, 1)[0]
    dn = x[x.spy < 0]
    bdn = np.polyfit(dn.spy, dn.fx, 1)[0]
    print(f"weekly {a}-{b}: corr {x.spy.corr(x.fx):.2f} beta {bt:.2f} downside-beta {bdn:.2f}")

# ---------------- episodes ----------------
print("\n=== T5 episodes: max drawdown inside each window, SPY in USD vs KRW (each its own peak->trough) ===")
eps = [("1997-07-01", "1998-12-31", "1997-98 KRW crash+rebound"), ("2000-03-01", "2003-03-31", "Dot-com"), ("2007-10-01", "2009-06-30", "GFC"),
       ("2009-03-01", "2009-12-31", "2009 rebound (KRW rally)"), ("2011-04-01", "2011-12-31", "2011"), ("2015-07-01", "2016-03-31", "2015-16"),
       ("2018-09-01", "2018-12-31", "2018Q4"), ("2020-02-01", "2020-04-30", "Covid"), ("2022-01-01", "2022-12-31", "2022"),
       ("2025-02-01", "2025-05-31", "2025 tariffs")]
def mdd(s):
    s = s.dropna(); dd = s / s.cummax() - 1; tr = dd.idxmin(); pk = s.loc[:tr].idxmax(); return dd.min(), pk, tr
for a, b, lab in eps:
    s = p.loc[a:b]
    du, pu, tu = mdd(s.spy); dk, pk_, tk = mdd(s.spy_krw)
    off = 1 - dk / du if du < -0.05 else np.nan
    print(f"{lab:28s} USD {fmt_pct(du):>7s} ({pu.date()}->{tu.date()})  KRW {fmt_pct(dk):>7s} ({pk_.date()}->{tk.date()})  FX over USD peak->trough {fmt_pct(s.fx.loc[tu]/s.fx.loc[pu]-1):>6s}  offset {fmt_pct(off,0) if off==off else '':>5s}")
for a, b, lab in [("2017-12-01", "2018-12-31", "BTC 2018"), ("2021-11-01", "2022-12-31", "BTC 2021-22"), ("2025-10-01", "2026-10-13", "BTC 2025-26")]:
    s = p.loc[a:b]
    du, pu, tu = mdd(s.btc); dk, pk_, tk = mdd(s.btc_krw)
    print(f"{lab:28s} USD {fmt_pct(du)}  KRW {fmt_pct(dk)}  ({pu.date()}->{tu.date()})")

# ---------------- FX-driven KRW losses ----------------
print("\n=== T6 worst 10 SPY-KRW months 1993+ with decomposition ===")
s = mr.loc["1993-02":E1, ["spy", "fx", "spy_krw"]].sort_values("spy_krw").head(10)
print(s.apply(lambda c: c.map(fmt_pct)).to_string())
s = mr.loc["2004-01":E1]
n_fx = ((s.spy > 0) & (s.spy_krw < -0.03)).sum()
print("months 2004+ with SPY USD up but SPY KRW < -3%:", n_fx, " ; 1993+:", ((mr.loc['1993-02':E1].spy > 0) & (mr.loc['1993-02':E1].spy_krw < -0.03)).sum())
# KRW appreciation shock tail
f = mr.fx.loc["1993-02":E1]
print("USD/KRW monthly change p1/p5/min:", fmt_pct(f.quantile(.01)), fmt_pct(f.quantile(.05)), fmt_pct(f.min()), f.idxmin().strftime("%Y-%m"))
print("FX vol monthly ann 2004+:", fmt_pct(mr.fx.loc['2004':E1].std() * np.sqrt(12)), " 2017-12+:", fmt_pct(mr.fx.loc['2017-12':E1].std() * np.sqrt(12)))
T1.to_csv("out_T1.csv", index=False); T2.to_csv("out_T2.csv", index=False)
pd.DataFrame(mixes).to_csv("mix_monthly.csv")
