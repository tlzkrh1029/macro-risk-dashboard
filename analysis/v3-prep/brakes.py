"""Section 3: loss-limit definitions in KRW terms.
(i) static sizing so that the historical 99th-pct monthly KRW loss = L (in-sample and expanding-window);
(ii) month-to-date brake: MTD KRW loss <= -L -> 50% of target until month end;
(iii) trailing-drawdown brake on the un-braked strategy NAV: DD <= -D -> 50%, restore when DD >= -D/2.
Whipsaw and missed-rebound diagnostics.
"""
import numpy as np, pandas as pd
from lib import simulate, summarize, fmt_pct, monthly
from sims import (p, R_ALL, tf_const, tf_eq_vt, tf_total_vt, tf_sleeve_vt, mix_base, run_eq, run_mx, fmt_row,
                  EQ_START, EQ_EVAL, MX_START, MX_EVAL, END, END_M)

pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30)

WINDOWS = [("2009-03-10", "2009-06-30", "2009 Q2 rebound"), ("2019-01-01", "2019-04-30", "2019-01..04"), ("2020-04-01", "2020-04-30", "2020-04"),
           ("2020-04-01", "2020-08-31", "2020-04..08"), ("2022-11-01", "2022-11-30", "2022-11"), ("2023-01-01", "2023-12-31", "2023"),
           ("2025-04-09", "2025-06-30", "2025-04-09..06")]


def window_stats(res, base, a, b):
    nb = base["nav"]; ns = res["nav"]
    a0 = nb.index[nb.index < pd.Timestamp(a)][-1]
    b0 = nb.index[nb.index <= pd.Timestamp(b)][-1]
    rb = nb.loc[b0] / nb.loc[a0] - 1
    rs = ns.loc[b0] / ns.loc[a0] - 1
    ex = res["w"].sum(axis=1).loc[a:b].mean()
    return rs, rb, ex


def brake_episodes(res, base):
    """Pair on/off events; for each, base-strategy return from brake execution to restore (or month end for MTD)."""
    out = []
    log = res["brake_log"]
    nb = base["nav"]
    i = 0
    while i < len(log):
        d_on, k = log[i]
        if k != "on":
            i += 1; continue
        d_off = log[i + 1][0] if i + 1 < len(log) and log[i + 1][1] == "off" else nb.index[-1]
        seg_b = nb.loc[d_on:d_off]
        seg_s = res["nav"].loc[d_on:d_off]
        out.append({"on": d_on.date(), "off": d_off.date(), "days": (d_off - d_on).days,
                    "base_ret": seg_b.iloc[-1] / seg_b.iloc[0] - 1, "braked_ret": seg_s.iloc[-1] / seg_s.iloc[0] - 1})
        i += 2 if (i + 1 < len(log) and log[i + 1][1] == "off") else 1
    return pd.DataFrame(out)


def describe(name, res, base, ev):
    s = summarize(res, ev, END_M)
    row = fmt_row(name, s)
    ep = brake_episodes(res, base) if res["brake_log"] else pd.DataFrame()
    if len(ep):
        ep = ep[pd.to_datetime(ep["on"]) >= pd.Timestamp(ev)]
    yrs = (pd.Timestamp(END) - pd.Timestamp(ev)).days / 365.25
    row["brakes/yr"] = f"{len(ep)/yrs:.2f}" if len(ep) else "0"
    if len(ep):
        row["cost%eps"] = fmt_pct((ep.braked_ret < ep.base_ret).mean(), 0)  # share of brake episodes where the brake cost money
        row["avg saved"] = fmt_pct((ep.braked_ret - ep.base_ret).mean())  # >0 means brake helped on average
    mr = monthly(res["nav"]).loc[ev:END_M]
    return row, ep, mr


def rebounds(name, res, base):
    r = {"strategy": name}
    for a, b, lab in WINDOWS:
        if pd.Timestamp(a) < res["nav"].index[0] + pd.Timedelta(days=40):
            continue
        rs, rb, ex = window_stats(res, base, a, b)
        r[lab] = f"{fmt_pct(rs-rb)} (x{ex:.2f})"
    return r


if __name__ == "__main__":
    for uni in ["eq", "mx", "gspc"]:
        if uni == "eq":
            runner, ev = run_eq, EQ_EVAL
            bases = {"B&H 100%": tf_const([1.0]), "VT15 ewma": tf_eq_vt(0.15, "ewma")}
            title = "SPY in KRW, eval 2004-01..2026-09"
            kw = {}
        elif uni == "mx":
            runner, ev = run_mx, MX_EVAL
            bases = {"const 85/15": tf_const(mix_base()), "total VT15 ewma": tf_total_vt(0.15, "ewma"),
                     "eqVT15+static crypto": tf_sleeve_vt(0.15, None, "ewma", crypto_static=True)}
            title = "85/15 SPY/crypto in KRW, eval 2017-12..2026-09"
            kw = {}
        else:
            runner, ev = run_eq, "1982-01"
            bases = {"B&H 100%": tf_const([1.0]), "VT15 ewma": tf_eq_vt(0.15, "ewma", "gspc_krw")}
            title = "^GSPC price in KRW (H.10 FX pre-2004), eval 1982-01..2026-09"
            kw = {"col": "gspc_krw", "start": "1981-06-01", "ev": "1982-01"}
        print(f"\n\n######## {title} ########")
        rows, rb_rows = [], []
        for bn, tf in bases.items():
            base, bs = runner(tf, **kw)
            rows.append(describe(bn, base, base, ev)[0])
            if uni != "gspc":
                rb_rows.append(rebounds(bn + " (vs itself)", base, base))
            for L in [0.05, 0.08, 0.10]:
                for chk in ["daily", "weekly"]:
                    res, _ = runner(tf, brake={"kind": "mtd", "L": L, "cut": 0.5}, brake_check=chk, **kw)
                    row, ep, _ = describe(f"{bn} + MTD {int(L*100)}% {chk}", res, base, ev)
                    rows.append(row)
                    if chk == "daily" and uni != "gspc":
                        rb_rows.append(rebounds(f"{bn} + MTD {int(L*100)}% daily", res, base))
            for D in [0.10, 0.15, 0.20]:
                res, _ = runner(tf, brake={"kind": "dd", "D": D, "cut": 0.5, "recover_frac": 0.5}, base_nav=base["nav"], brake_check="weekly", **kw)
                row, ep, _ = describe(f"{bn} + DD {int(D*100)}% wk", res, base, ev)
                rows.append(row)
                if uni != "gspc":
                    rb_rows.append(rebounds(f"{bn} + DD {int(D*100)}% wk", res, base))
                if uni == "mx" and bn == "const 85/15" and D == 0.15:
                    print("DD15 episodes on const 85/15:"); print(ep.to_string(index=False))
                if uni == "eq" and bn == "B&H 100%" and D == 0.15:
                    print("DD15 episodes on SPY-KRW B&H:"); print(ep.to_string(index=False))
        print(pd.DataFrame(rows).to_string(index=False))
        if rb_rows:
            print("\nRebound windows: strategy return minus its un-braked base over the window (avg exposure)")
            print(pd.DataFrame(rb_rows).to_string(index=False))

    # ---------------- (i) static loss-limit sizing ----------------
    print("\n\n######## (i) static sizing: w = L / |q99 monthly KRW loss| ########")
    mm = p[["spy_krw", "btc_krw", "eth_krw"]].resample("ME").last().pct_change(fill_method=None)
    b = mix_base()
    # proxy ETH by BTC before ETH exists (for expanding-window history only)
    eth_px = mm["eth_krw"].fillna(mm["btc_krw"])
    mix_m = b[0] * mm["spy_krw"] + b[1] * mm["btc_krw"] + b[2] * eth_px
    eq_m = mm["spy_krw"]
    out = []
    for nm, ser, a0, runner, base_tf, ev in [("SPY KRW", eq_m, "1993-02", run_eq, lambda w: tf_const([w]), EQ_EVAL),
                                              ("85/15 KRW", mix_m, "2014-10", run_mx, lambda w: tf_const(w * mix_base()), MX_EVAL)]:
        full = ser.loc[ev:END_M]
        q99 = -np.quantile(full, 0.01); q95 = -np.quantile(full, 0.05)
        for L in [0.05, 0.08, 0.10, 0.15]:
            w = min(1.0, L / q99)
            res, s = runner(base_tf(w))
            exc = (monthly(res["nav"]).loc[ev:END_M] < -L).sum()
            # expanding-window (no look-ahead): monthly w_t from all history before t (min 36 months)
            hist = ser.loc[a0:]
            wt = {}
            for d in hist.index:
                past = hist.loc[:d - pd.offsets.MonthEnd(1)].dropna() if False else hist.loc[:d].iloc[:-1].dropna()
                if len(past) >= 36:
                    wt[d] = min(1.0, L / -np.quantile(past, 0.01))
            wt = pd.Series(wt)
            wd = wt.reindex(p.index, method="ffill").shift(1)  # w set at month-end applies next month
            if nm == "SPY KRW":
                tfx = (lambda wd: (lambda d: np.array([wd.loc[d] if wd.loc[d] == wd.loc[d] else 1.0])))(wd)
            else:
                tfx = (lambda wd: (lambda d: (wd.loc[d] if wd.loc[d] == wd.loc[d] else 1.0) * mix_base()))(wd)
            res2, s2 = runner(tfx)
            exc2 = (monthly(res2["nav"]).loc[ev:END_M] < -L).sum()
            out.append({"universe": nm, "L": fmt_pct(L, 0), "in-sample q99": fmt_pct(q99), "w (in-sample)": f"{w:.2f}",
                        "realised worst": fmt_pct(s["worst_m"]), "months > L": int(exc), "maxDD": fmt_pct(s["maxdd"]),
                        "expanding w range": f"{wt.loc[ev:].min():.2f}-{wt.loc[ev:].max():.2f}", "exp worst": fmt_pct(s2["worst_m"]),
                        "exp months>L": int(exc2), "exp avg expo": f"{s2['avg_expo']:.2f}", "exp maxDD": fmt_pct(s2["maxdd"])})
        # stability of q99 estimate over time
        st = {}
        for y in ["2008-12", "2012-12", "2016-12", "2019-12", "2021-12", "2023-12", "2026-09"]:
            past = ser.loc[a0:y].dropna()
            if len(past) < 12:
                continue
            st[y] = f"{fmt_pct(-np.quantile(past, 0.01))} (n={len(past)})"
        print(nm, "q99 monthly loss estimated with data through:", st)
    print(pd.DataFrame(out).to_string(index=False))
