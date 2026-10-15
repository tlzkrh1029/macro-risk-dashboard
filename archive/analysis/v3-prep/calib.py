"""Sections 4-5: hybrid calibration. For a KRW monthly loss tolerance L at 95%/99%, find the annual vol target T
(EWMA, NW-corrected, weekly Friday signal, executed next US day close, cap 1.0) whose realised monthly KRW loss
quantile equals L. Compare with the normal-theory rule and with static sizing at the same L.
"""
import numpy as np, pandas as pd
from lib import fmt_pct, monthly
from sims import (p, R_ALL, tf_const, tf_eq_vt, tf_total_vt, tf_sleeve_vt, mix_base, run_eq, run_mx, EQ_EVAL, MX_EVAL, END, END_M)

pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
GRID = np.round(np.arange(0.04, 0.401, 0.01), 3)


def metrics(res, s):
    return {"p5": -s["p5"], "p1": -s["p1"], "worst": -s["worst_m"], "worst_date": s["worst_m_date"], "maxdd": -s["maxdd"],
            "cagr": s["cagr"], "expo": s["avg_expo"], "turn": s["turnover"], "vol": s["vol"]}


def sweep(kind):
    rows = {}
    for T in GRID:
        if kind == "eq":
            res, s = run_eq(tf_eq_vt(T, "ewma"))
        elif kind == "gspc":
            res, s = run_eq(tf_eq_vt(T, "ewma", "gspc_krw"), col="gspc_krw", start="1981-06-01", ev="1982-01")
        elif kind == "mx":
            res, s = run_mx(tf_total_vt(T, "ewma"))
        rows[T] = metrics(res, s)
    return pd.DataFrame(rows).T


def solve(df, col, L):
    """Smallest T where loss quantile reaches L (linear interpolation on the monotone envelope)."""
    y = df[col].astype(float).cummax()  # monotone envelope in T
    x = df.index.values.astype(float)
    if y.iloc[-1] < L:
        return np.nan  # cap binds: even fully invested the loss quantile is below L
    if y.iloc[0] >= L:
        return x[0]
    k = np.argmax(y.values >= L)
    x0, x1, y0, y1 = x[k - 1], x[k], y.values[k - 1], y.values[k]
    return x0 + (L - y0) * (x1 - x0) / (y1 - y0) if y1 > y0 else x1


if __name__ == "__main__":
    sw = {k: sweep(k) for k in ["eq", "mx", "gspc"]}
    for k, df in sw.items():
        print(f"\n=== sweep {k} (selected T) ===")
        d = df.loc[[0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]].copy()
        print(d.assign(**{c: d[c].map(lambda v: fmt_pct(v)) for c in ["p5", "p1", "worst", "maxdd", "cagr", "vol"]}).to_string())
        df.to_csv(f"sweep_{k}.csv")
    # empirical multipliers k = loss quantile / (T/sqrt12) and vs realised vol
    print("\n=== empirical tail multipliers at T=15% (loss quantile / monthly target vol; / monthly realised vol) ===")
    for k, df in sw.items():
        r = df.loc[0.15]
        print(k, f"k95 vs target {r.p5/(0.15/np.sqrt(12)):.2f}  k99 vs target {r.p1/(0.15/np.sqrt(12)):.2f}  worst vs target {r.worst/(0.15/np.sqrt(12)):.2f}",
              f"| vs realised vol: k95 {r.p5/(r.vol/np.sqrt(12)):.2f} k99 {r.p1/(r.vol/np.sqrt(12)):.2f} worst {r.worst/(r.vol/np.sqrt(12)):.2f}")
    # static loss quantiles (for comparison at the same L)
    eqm = monthly(p["spy_krw"].loc["2003-12-31":END]).loc[EQ_EVAL:END_M]
    b = mix_base()
    mm = p[["spy_krw", "btc_krw", "eth_krw"]].resample("ME").last().pct_change(fill_method=None).loc[MX_EVAL:END_M]
    mxm = b[0] * mm.spy_krw + b[1] * mm.btc_krw + b[2] * mm.eth_krw
    out = []
    for L in [0.05, 0.08, 0.10, 0.15]:
        for conf, col, z in [("95%", "p5", 1.645), ("99%", "p1", 2.326)]:
            row = {"L": fmt_pct(L, 0), "conf": conf}
            for k, lab, static_ser, runner, tf_static in [("eq", "Equity (SPY KRW 04-26)", eqm, run_eq, lambda w: tf_const([w])),
                                                          ("mx", "85/15 (KRW 17-26)", mxm, run_mx, lambda w: tf_const(w * mix_base()))]:
                df = sw[k]
                T = solve(df, col, L)
                if np.isnan(T):
                    res, s = (run_eq(tf_const([1.0])) if k == "eq" else run_mx(tf_const(mix_base())))
                    m = metrics(res, s)
                    row[f"{lab} T"] = "cap binds (100%)"
                    row[f"{lab} worst/maxDD"] = f"{fmt_pct(-m['worst'])} / {fmt_pct(-m['maxdd'])}"
                else:
                    res, s = (run_eq(tf_eq_vt(T, "ewma")) if k == "eq" else run_mx(tf_total_vt(T, "ewma")))
                    m = metrics(res, s)
                    row[f"{lab} T"] = f"{100*T:.1f}%"
                    row[f"{lab} worst/maxDD"] = f"{fmt_pct(-m['worst'])} ({m['worst_date']}) / {fmt_pct(-m['maxdd'])}"
                    row[f"{lab} CAGR/expo"] = f"{fmt_pct(m['cagr'])} / {m['expo']:.2f}"
                # normal-theory target from full-sample vol of the base
                row[f"{lab} normal-rule T"] = f"{100*L*np.sqrt(12)/z:.1f}%"
                # static at same L
                q = -np.quantile(static_ser, 0.05 if conf == "95%" else 0.01)
                w = min(1.0, L / q)
                rs, ss = runner(tf_static(w))
                row[f"{lab} static w: worst/maxDD/CAGR"] = f"w={w:.2f}: {fmt_pct(ss['worst_m'])} / {fmt_pct(ss['maxdd'])} / {fmt_pct(ss['cagr'])}"
            # 1982+ GSPC cross-check
            Tg = solve(sw["gspc"], col, L)
            if np.isnan(Tg):
                row["GSPC-KRW 82-26 T"] = "cap binds"
            else:
                rg, sg = run_eq(tf_eq_vt(Tg, "ewma", "gspc_krw"), col="gspc_krw", start="1981-06-01", ev="1982-01")
                row["GSPC-KRW 82-26 T"] = f"{100*Tg:.1f}% (worst {fmt_pct(sg['worst_m'])})"
            out.append(row)
    C = pd.DataFrame(out)
    print("\n=== CALIBRATION TABLE ===")
    for c in C.columns:
        pass
    print(C.to_string(index=False))
    C.to_csv("calibration_table.csv", index=False)

    # hybrid + hard brake (daily MTD brake at L) for the 99% calibration
    print("\n=== hybrid VT(T from L@99%) with and without a daily MTD brake at L (50% cut) ===")
    rows = []
    for L in [0.05, 0.08, 0.10]:
        for k in ["eq", "mx"]:
            T = solve(sw[k], "p1", L)
            if np.isnan(T):
                continue
            tf = tf_eq_vt(T, "ewma") if k == "eq" else tf_total_vt(T, "ewma")
            runner = run_eq if k == "eq" else run_mx
            for br in [None, {"kind": "mtd", "L": L, "cut": 0.5}, {"kind": "mtd", "L": L * 0.6, "cut": 0.5}]:
                res, s = runner(tf, brake=br, brake_check="daily")
                ev = EQ_EVAL if k == "eq" else MX_EVAL
                mr = monthly(res["nav"]).loc[ev:END_M]
                rows.append({"univ": k, "L": fmt_pct(L, 0), "T": f"{100*T:.1f}%", "brake": "none" if br is None else f"MTD {fmt_pct(br['L'])} daily",
                             "p1": fmt_pct(s["p1"]), "worst": f"{fmt_pct(s['worst_m'])} {s['worst_m_date']}", "months>L": int((mr < -L).sum()),
                             "maxDD": fmt_pct(s["maxdd"]), "CAGR": fmt_pct(s["cagr"]), "expo": f"{s['avg_expo']:.2f}", "turn": f"{s['turnover']:.2f}"})
    print(pd.DataFrame(rows).to_string(index=False))

    # 2-D lookup for the split design: equity VT T_e + static crypto weight c (BTC:ETH 2:1)
    print("\n=== split design lookup: equity VT T_e (on 1-c of capital) + static crypto c, KRW 2017-12..2026-09 ===")
    rows = []
    for c in [0.0, 0.05, 0.10, 0.15, 0.20]:
        for Te in [0.10, 0.12, 0.15, 0.18]:
            res, s = run_mx(tf_sleeve_vt(Te, None, "ewma", eq=1 - c, crypto_static=True))
            rows.append({"crypto c": fmt_pct(c, 0), "T_e": fmt_pct(Te, 0), "p5": fmt_pct(s["p5"]), "p1": fmt_pct(s["p1"]),
                         "worst": f"{fmt_pct(s['worst_m'])} {s['worst_m_date']}", "worst3m": fmt_pct(s["worst_3m"]), "maxDD": fmt_pct(s["maxdd"]),
                         "CAGR": fmt_pct(s["cagr"]), "vol": fmt_pct(s["vol"]), "expo": f"{s['avg_expo']:.2f}"})
    print(pd.DataFrame(rows).to_string(index=False))
