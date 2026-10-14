"""Section 2: volatility-target definitions (KRW terms), weekly Friday signal, executed next US trading day close.
Also provides builders reused by brakes.py / calib.py.
"""
import sys
import numpy as np, pandas as pd
from lib import panel, simulate, summarize, vol_est, fmt_pct, monthly, tails

p = panel()
R_ALL = p[["spy_krw", "btc_krw", "eth_krw", "gspc_krw", "spy", "gspc"]].pct_change(fill_method=None)
EQ_START, EQ_EVAL = "2003-09-01", "2004-01"
MX_START, MX_EVAL = "2017-11-30", "2017-12"
END, END_M = "2026-09-30", "2026-09"


def series_vol(r, est):
    return vol_est(r, est)


def tf_const(w):
    w = np.asarray(w, float)
    return lambda d: w


def tf_eq_vt(T, est, col="spy_krw"):
    v = series_vol(R_ALL[col], est)
    return lambda d: np.array([min(1.0, T / v.loc[d])])


def mix_base(eq=0.85, btc_share=2 / 3):
    c = 1 - eq
    return np.array([eq, c * btc_share, c * (1 - btc_share)])


def tf_sleeve_vt(Te, Tc, est, eq=0.85, btc_share=2 / 3, crypto_static=False):
    ve = series_vol(R_ALL["spy_krw"], est)
    rc = btc_share * R_ALL["btc_krw"].fillna(0) + (1 - btc_share) * R_ALL["eth_krw"].fillna(0)
    vc = series_vol(rc, est)
    b = mix_base(eq, btc_share)

    def f(d):
        se = min(1.0, Te / ve.loc[d]) if Te else 1.0
        sc = 1.0 if crypto_static else min(1.0, Tc / vc.loc[d])
        return np.array([b[0] * se, b[1] * sc, b[2] * sc])
    return f


def tf_total_vt(Tp, est, eq=0.85, btc_share=2 / 3):
    b = mix_base(eq, btc_share)
    rb = b[0] * R_ALL["spy_krw"].fillna(0) + b[1] * R_ALL["btc_krw"].fillna(0) + b[2] * R_ALL["eth_krw"].fillna(0)
    vp = series_vol(rb, est)
    return lambda d: b * min(1.0, Tp / vp.loc[d])


def run_eq(tf, col="spy_krw", start=EQ_START, ev=EQ_EVAL, **kw):
    res = simulate(R_ALL[[col]], tf, p, start, END, **kw)
    return res, summarize(res, ev, END_M)


def run_mx(tf, start=MX_START, ev=MX_EVAL, **kw):
    res = simulate(R_ALL[["spy_krw", "btc_krw", "eth_krw"]], tf, p, start, END, **kw)
    return res, summarize(res, ev, END_M)


def calib(res, T, ev):
    """Realised-vs-target: full-sample vol from weekly and monthly returns, and rolling 13-week realised vol / target."""
    nav = res["nav"].loc[ev:END]
    wk = nav.resample("W-FRI").last().pct_change(fill_method=None).dropna()
    mo = nav.resample("ME").last().pct_change(fill_method=None).dropna()
    rv13 = wk.rolling(13).std() * np.sqrt(52)
    ratio = (rv13 / T).dropna()
    return {"rv_wk": wk.std() * np.sqrt(52), "rv_mo": mo.std() * np.sqrt(12),
            "r13_med": ratio.median(), "r13_p90": ratio.quantile(0.9), "r13_max": ratio.max(), "pct_r13_gt1.5": (ratio > 1.5).mean()}


def fmt_row(name, s, c=None, T=None):
    out = {"strategy": name, "p5": fmt_pct(s["p5"]), "p1": fmt_pct(s["p1"]), "worst_m": f"{fmt_pct(s['worst_m'])} {s['worst_m_date']}",
           "worst_3m": fmt_pct(s["worst_3m"]), "maxDD": fmt_pct(s["maxdd"]), "CAGR": fmt_pct(s["cagr"]), "vol_m": fmt_pct(s["vol"]),
           "ret/vol": f"{s['cagr']/s['vol']:.2f}", "avg_expo": f"{s['avg_expo']:.2f}", "turn/yr": f"{s['turnover']:.2f}"}
    if c is not None:
        out.update({"vol_wk": fmt_pct(c["rv_wk"]), "r13/T med": f"{c['r13_med']:.2f}", "r13/T p90": f"{c['r13_p90']:.2f}",
                    "r13/T max": f"{c['r13_max']:.2f}", ">1.5T": fmt_pct(c["pct_r13_gt1.5"], 0)})
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30)
    # ---------------- equity only, SPY KRW 2004+ ----------------
    rows = []
    res, s = run_eq(tf_const([1.0])); rows.append(fmt_row("B&H 100%", s))
    for T in [0.12, 0.15, 0.18]:
        for est in ["20d", "60d", "ewma", "max"]:
            res, s = run_eq(tf_eq_vt(T, est)); c = calib(res, T, EQ_EVAL)
            rows.append(fmt_row(f"VT {int(T*100)}% {est}", s, c, T))
    # USD-measured vol (same KRW outcome), and Friday-close execution
    for T in [0.15]:
        res, s = run_eq(tf_eq_vt(T, "ewma", col="spy")); rows.append(fmt_row(f"VT {int(T*100)}% ewma on USD vol (KRW P&L)", s))
        res, s = run_eq(tf_eq_vt(T, "ewma"), exec_lag=0); c = calib(res, T, EQ_EVAL); rows.append(fmt_row(f"VT {int(T*100)}% ewma, Fri-close exec", s, c))
        res, s = run_eq(tf_eq_vt(T, "ewma"), band=0.10); c = calib(res, T, EQ_EVAL); rows.append(fmt_row(f"VT {int(T*100)}% ewma, 10%p band", s, c))
        res, s = run_eq(tf_eq_vt(T, "ewma"), cash="usd", fx=p["fx"], irx=p["irx"]); rows.append(fmt_row(f"VT {int(T*100)}% ewma, cash in USD T-bills", s))
    T2 = pd.DataFrame(rows)
    print("=== S2a equity-only (SPY in KRW), eval 2004-01..2026-09, weekly signal, exec next US day close ===")
    print(T2.to_string(index=False))
    # constant exposure with same average exposure as VT15 ewma
    res, s = run_eq(tf_eq_vt(0.15, "ewma")); ae = s["avg_expo"]
    res2, s2 = run_eq(tf_const([ae])); print(f"\nconstant {ae:.2f} (matches VT15 ewma avg exposure):", fmt_row("const", s2))
    # ^GSPC KRW 1982+ cross-check (price only, H.10 FX pre-2004)
    print("\n=== S2b cross-check ^GSPC price in KRW, eval 1982-01..2026-09 ===")
    rows = []
    for nm, tf in [("B&H", tf_const([1.0])), ("VT 12% ewma", tf_eq_vt(0.12, "ewma", "gspc_krw")), ("VT 15% ewma", tf_eq_vt(0.15, "ewma", "gspc_krw")),
                   ("VT 18% ewma", tf_eq_vt(0.18, "ewma", "gspc_krw"))]:
        res, s = run_eq(tf, col="gspc_krw", start="1981-06-01", ev="1982-01"); rows.append(fmt_row(nm, s))
    print(pd.DataFrame(rows).to_string(index=False))

    # ---------------- mix 85/15 (BTC:ETH 2:1) 2017-12+ ----------------
    rows = []
    res, s = run_mx(tf_const(mix_base())); rows.append(fmt_row("const 85/15 (weekly rebal)", s))
    for Te in [0.12, 0.15, 0.18]:
        for Tc in [0.30, 0.40, 0.50]:
            res, s = run_mx(tf_sleeve_vt(Te, Tc, "ewma")); rows.append(fmt_row(f"sleeve VT eq {int(Te*100)} / cr {int(Tc*100)} ewma", s))
    for Te in [0.12, 0.15, 0.18]:
        res, s = run_mx(tf_sleeve_vt(Te, None, "ewma", crypto_static=True)); rows.append(fmt_row(f"eq VT {int(Te*100)} + static crypto 15%", s))
    for Tp in [0.10, 0.12, 0.15, 0.18]:
        for est in ["60d", "ewma"]:
            res, s = run_mx(tf_total_vt(Tp, est)); c = calib(res, Tp, MX_EVAL)
            rows.append(fmt_row(f"total VT {int(Tp*100)}% {est}", s, c, Tp))
    T3 = pd.DataFrame(rows)
    print("\n=== S2c 85/15 SPY/crypto (BTC:ETH 2:1), KRW, eval 2017-12..2026-09 ===")
    print(T3.to_string(index=False))
    # crypto vol-scaling vs static at equal average exposure
    res, s = run_mx(tf_sleeve_vt(0.15, 0.40, "ewma"))
    wc = res["w"][["btc_krw", "eth_krw"]].sum(axis=1).loc[MX_EVAL:].mean(); we = res["w"]["spy_krw"].loc[MX_EVAL:].mean()
    print(f"\nsleeve VT 15/40 avg weights: equity {we:.3f} crypto {wc:.3f}")
    res2, s2 = run_mx(tf_sleeve_vt(0.15, None, "ewma", eq=0.85, crypto_static=True))
    # equal-average crypto static: scale crypto base to wc
    b = np.array([0.85, wc * 2 / 3, wc / 3])
    ve = vol_est(R_ALL["spy_krw"], "ewma")
    tfx = lambda d: np.array([b[0] * min(1, 0.15 / ve.loc[d]), b[1], b[2]])
    res3, s3 = run_mx(tfx)
    print("crypto vol-scaled 40%:", fmt_row("", s))
    print(f"crypto static {wc:.3f}:  ", fmt_row("", s3))
    T2.to_csv("out_S2a.csv", index=False); T3.to_csv("out_S2c.csv", index=False)
