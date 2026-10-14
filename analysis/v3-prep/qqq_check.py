"""Quick check: does the T ~ 2 x L99 mapping hold for a tech-heavy equity sleeve (QQQ in KRW)?"""
import numpy as np, pandas as pd
from lib import simulate, summarize, vol_est, fmt_pct, tails, monthly
from sims import p
SP = "/tmp/claude-0/-home-user-macro-risk-dashboard/157fe875-3352-5fc5-b8c1-31e705d9a652/scratchpad"
q = pd.read_csv(f"{SP}/data/krw_crypto.csv", index_col=0, parse_dates=True)["QQQ"].dropna().loc[:"2026-10-13"]
q = q.reindex(p.index).ffill(limit=6)
qk = q * p["fx"]
R = pd.DataFrame({"qqq_krw": qk.pct_change(fill_method=None)})
t = tails(monthly(qk.loc["2003-12-31":"2026-09-30"]).loc["2004-01":"2026-09"], qk.loc["2004-01-01":"2026-09-30"] / qk.loc["2004-01-02"])
print("QQQ KRW B&H 2004-26:", {k: (fmt_pct(v) if isinstance(v, float) and k not in ("k95", "k99") else v) for k, v in t.items()})
v = vol_est(R["qqq_krw"], "ewma")
for T in [0.10, 0.15, 0.16, 0.20]:
    res = simulate(R, lambda d, T=T: np.array([min(1.0, T / v.loc[d])]), p, "2003-09-01", "2026-09-30")
    s = summarize(res, "2004-01", "2026-09")
    print(f"QQQ KRW VT{int(T*100)}: p1 {fmt_pct(s['p1'])} (=> k99 {-s['p1']/(T/np.sqrt(12)):.2f}, p1/T {-s['p1']/T:.2f}) worst {fmt_pct(s['worst_m'])} {s['worst_m_date']} maxDD {fmt_pct(s['maxdd'])} CAGR {fmt_pct(s['cagr'])} expo {s['avg_expo']:.2f}")
