"""Extra robustness: rebound windows vs buy-and-hold, KRW- vs USD-measured vol, cash currency, sub-period k multipliers."""
import numpy as np, pandas as pd
from lib import fmt_pct, monthly, vol_est
from sims import (p, R_ALL, tf_const, tf_eq_vt, tf_total_vt, tf_sleeve_vt, mix_base, run_eq, run_mx, EQ_EVAL, MX_EVAL, END, END_M)
from brakes import WINDOWS, window_stats

pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)

# 1) rebound windows relative to buy-and-hold
for uni in ["eq", "mx"]:
    if uni == "eq":
        bh, _ = run_eq(tf_const([1.0]))
        strats = {"static w=0.76 (L=8%@99%)": run_eq(tf_const([0.76]))[0], "VT15 ewma": run_eq(tf_eq_vt(0.15, "ewma"))[0],
                  "VT17 ewma (L=8%@99%)": run_eq(tf_eq_vt(0.172, "ewma"))[0],
                  "B&H + MTD 8% daily": run_eq(tf_const([1.0]), brake={"kind": "mtd", "L": 0.08, "cut": 0.5})[0],
                  "B&H + DD 15% weekly": run_eq(tf_const([1.0]), brake={"kind": "dd", "D": 0.15, "cut": 0.5}, base_nav=bh["nav"], brake_check="weekly")[0]}
    else:
        bh, _ = run_mx(tf_const(mix_base()))
        strats = {"static w=0.78 (L=8%@99%)": run_mx(tf_const(0.78 * mix_base()))[0], "total VT15 ewma": run_mx(tf_total_vt(0.15, "ewma"))[0],
                  "total VT16 (L=8%@99%)": run_mx(tf_total_vt(0.16, "ewma"))[0],
                  "eqVT15 + static crypto 10%": run_mx(tf_sleeve_vt(0.15, None, "ewma", eq=0.90, crypto_static=True))[0],
                  "const + MTD 8% daily": run_mx(tf_const(mix_base()), brake={"kind": "mtd", "L": 0.08, "cut": 0.5})[0],
                  "const + DD 15% weekly": run_mx(tf_const(mix_base()), brake={"kind": "dd", "D": 0.15, "cut": 0.5}, base_nav=bh["nav"], brake_check="weekly")[0]}
    rows = []
    r0 = {"strategy": "B&H return in window"}
    for a, b, lab in WINDOWS:
        if pd.Timestamp(a) < bh["nav"].index[0] + pd.Timedelta(days=40):
            continue
        rs, rb, ex = window_stats(bh, bh, a, b)
        r0[lab] = fmt_pct(rb)
    rows.append(r0)
    for nm, res in strats.items():
        r = {"strategy": nm}
        for a, b, lab in WINDOWS:
            if pd.Timestamp(a) < bh["nav"].index[0] + pd.Timedelta(days=40):
                continue
            rs, rb, ex = window_stats(res, bh, a, b)
            r[lab] = f"{fmt_pct(rs)} (x{ex:.2f})"
        rows.append(r)
    print(f"\n=== rebound windows ({uni}): strategy return in window (avg exposure) ===")
    print(pd.DataFrame(rows).to_string(index=False))

# 2) KRW- vs USD-measured vol for the equity target, several T and samples
print("\n=== equity VT: vol measured on SPY in KRW vs in USD (P&L always in KRW) ===")
rows = []
for T in [0.10, 0.12, 0.15, 0.18]:
    for col_sig, lab in [("spy_krw", "KRW vol"), ("spy", "USD vol")]:
        res, s = run_eq(tf_eq_vt(T, "ewma", col_sig))
        rows.append({"T": fmt_pct(T, 0), "signal": lab, "p1": fmt_pct(s["p1"]), "worst": f"{fmt_pct(s['worst_m'])} {s['worst_m_date']}", "maxDD": fmt_pct(s["maxdd"]),
                     "CAGR": fmt_pct(s["cagr"]), "expo": f"{s['avg_expo']:.2f}", "turn": f"{s['turnover']:.2f}"})
print(pd.DataFrame(rows).to_string(index=False))

# 3) cash currency for de-risked money
print("\n=== cash currency (equity VT15 ewma and mix total VT15) ===")
for nm, runner, tf in [("eq VT15", run_eq, tf_eq_vt(0.15, "ewma")), ("mx total VT15", run_mx, tf_total_vt(0.15, "ewma")),
                       ("eq VT10", run_eq, tf_eq_vt(0.10, "ewma"))]:
    for cash in ["krw", "usd"]:
        kw = {"cash": cash} if cash == "krw" else {"cash": "usd", "fx": p["fx"], "irx": p["irx"]}
        res, s = runner(tf, **kw)
        print(f"{nm:14s} cash={cash}: p5 {fmt_pct(s['p5'])} p1 {fmt_pct(s['p1'])} worst {fmt_pct(s['worst_m'])} {s['worst_m_date']} maxDD {fmt_pct(s['maxdd'])} CAGR {fmt_pct(s['cagr'])}")

# 4) stability of the tail multiplier k99 = |p1| / (T/sqrt12) across sub-periods
print("\n=== sub-period stability of tail multipliers for VT15 ewma ===")
for nm, runner, tf, splits in [("eq", run_eq, tf_eq_vt(0.15, "ewma"), [("2004-01", "2014-12"), ("2015-01", "2026-09")]),
                               ("mx", run_mx, tf_total_vt(0.15, "ewma"), [("2017-12", "2021-12"), ("2022-01", "2026-09")])]:
    res, s = runner(tf)
    mr = monthly(res["nav"])
    for a, b in splits:
        x = mr.loc[a:b]
        m = 0.15 / np.sqrt(12)
        print(f"{nm} {a}..{b}: n={len(x)} p5 {fmt_pct(np.quantile(x,0.05))} p1 {fmt_pct(np.quantile(x,0.01))} worst {fmt_pct(x.min())} "
              f"k95 {-np.quantile(x,0.05)/m:.2f} k99 {-np.quantile(x,0.01)/m:.2f} kworst {-x.min()/m:.2f} realised vol {fmt_pct(x.std()*np.sqrt(12))}")

# 5) current readings (as of last Friday close 2026-10-09) for the recommended settings
print("\n=== current readings, Friday 2026-10-09 close ===")
d = pd.Timestamp("2026-10-09")
for nm, col in [("SPY KRW", "spy_krw"), ("SPY USD", "spy")]:
    print(nm, "EWMA vol", fmt_pct(vol_est(R_ALL[col], "ewma").loc[d]), " 60d", fmt_pct(vol_est(R_ALL[col], "60d").loc[d]))
b = mix_base()
rc = 2 / 3 * R_ALL["btc_krw"].fillna(0) + 1 / 3 * R_ALL["eth_krw"].fillna(0)
print("crypto sleeve 2:1 KRW EWMA vol", fmt_pct(vol_est(rc, "ewma").loc[d]), " 60d", fmt_pct(vol_est(rc, "60d").loc[d]))
rb = b[0] * R_ALL["spy_krw"].fillna(0) + b[1] * R_ALL["btc_krw"].fillna(0) + b[2] * R_ALL["eth_krw"].fillna(0)
vp = vol_est(rb, "ewma").loc[d]
print("85/15 portfolio KRW EWMA vol", fmt_pct(vp), "-> total VT15 scale", f"{min(1, 0.15/vp):.2f}", " VT16", f"{min(1,0.16/vp):.2f}", " VT10", f"{min(1,0.10/vp):.2f}")
for T in [0.09, 0.12, 0.15, 0.17]:
    print(f"equity VT{int(T*100)} exposure now: {min(1, T / vol_est(R_ALL['spy_krw'], 'ewma').loc[d]):.2f}")
# MTD as of 2026-10-13 for 85/15 const
nav = (1 + rb.loc["2026-09-30":"2026-10-13"]).cumprod()
print("85/15 const MTD KRW through 2026-10-13:", fmt_pct(nav.iloc[-1] / nav.iloc[0] - 1))
