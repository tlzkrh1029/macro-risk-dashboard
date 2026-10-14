"""Shared helpers: returns, tail stats, and a weekly-rebalance sizing simulator on a calendar-day grid.

Conventions
- Calendar-day grid. SPY/FX are forward-filled on non-trading days (0 return); crypto trades 7 days.
- All variances annualised x365 on the calendar grid (sum of daily variances per year).
- KRW value = USD value x KRW=X (KRW per USD).
- Cash earns 0% in KRW unless cash='usd' (then cash is USD T-bills (^IRX) valued in KRW).
- Signals use data through the signal-date close only; trades execute at the close of the next US trading
  day after the signal (Friday signal -> Monday close) unless exec_lag=0 (Friday close).
"""
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ANN = 365.0


def panel():
    return pd.read_pickle(os.path.join(HERE, "panel.pkl"))


def rets(p, cols):
    return p[cols].pct_change(fill_method=None)


def monthly(nav):
    m = nav.resample("ME").last()
    return m.pct_change(fill_method=None).dropna()


def tails(mr, nav=None):
    """mr: monthly simple returns (Series)."""
    mr = mr.dropna()
    r3 = (1 + mr).rolling(3).apply(np.prod, raw=True) - 1
    r12 = (1 + mr).rolling(12).apply(np.prod, raw=True) - 1
    out = {
        "n": len(mr),
        "p5": np.quantile(mr, 0.05),
        "p1": np.quantile(mr, 0.01),
        "worst_m": mr.min(),
        "worst_m_date": mr.idxmin().strftime("%Y-%m"),
        "worst_3m": r3.min(),
        "worst_3m_end": r3.idxmin().strftime("%Y-%m") if r3.notna().any() else "",
        "worst_12m": r12.min() if r12.notna().any() else np.nan,
        "vol": mr.std() * np.sqrt(12),
        "cagr": (1 + mr).prod() ** (12 / len(mr)) - 1,
    }
    out["k95"] = -out["p5"] / (out["vol"] / np.sqrt(12))
    out["k99"] = -out["p1"] / (out["vol"] / np.sqrt(12))
    if nav is not None:
        out["maxdd"] = (nav / nav.cummax() - 1).min()
    return out


def fmt_pct(x, d=1):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100*x:.{d}f}%"


# ---------------- volatility estimators (past data only) ----------------
def vol_window(r, days, nw=True):
    """Rolling realised vol over `days` calendar days with optional Newey-West lag-1 term
    (corrects for the asynchronous Yahoo FX close vs US equity close)."""
    r = r.fillna(0.0)
    v0 = (r * r).rolling(days, min_periods=max(10, days // 3)).mean()
    if nw:
        v1 = (r * r.shift(1)).rolling(days, min_periods=max(10, days // 3)).mean()
        v = (v0 + 2 * v1).clip(lower=0.5 * v0)
    else:
        v = v0
    return np.sqrt(v * ANN)


def vol_ewma(r, lam=0.94 ** (252 / 365), nw=True):
    r = r.fillna(0.0)
    v0 = (r * r).ewm(alpha=1 - lam, adjust=False).mean()
    if nw:
        v1 = (r * r.shift(1).fillna(0)).ewm(alpha=1 - lam, adjust=False).mean()
        v = (v0 + 2 * v1).clip(lower=0.5 * v0)
    else:
        v = v0
    return np.sqrt(v * ANN)


def vol_est(r, kind):
    if kind == "20d":
        return vol_window(r, 28)
    if kind == "60d":
        return vol_window(r, 84)
    if kind == "ewma":
        return vol_ewma(r)
    if kind == "max":  # conservative: max(20d, 60d)
        return np.maximum(vol_window(r, 28), vol_window(r, 84))
    raise ValueError(kind)


def cov_window(R, days):
    """Rolling covariance (calendar days) with NW lag-1 correction; returns dict date->matrix lazily."""
    R = R.fillna(0.0)
    cols = R.columns
    n = len(cols)
    out = {}
    for i in range(n):
        for j in range(i, n):
            a, b = R[cols[i]], R[cols[j]]
            c0 = (a * b).rolling(days, min_periods=max(10, days // 3)).mean()
            c1 = ((a * b.shift(1)) + (a.shift(1) * b)).rolling(days, min_periods=max(10, days // 3)).mean()
            out[(i, j)] = (c0 + c1) * ANN
    return out, cols


# ---------------- simulator ----------------
def signal_dates(p, start, end):
    """Last US trading day of each week (Friday, or Thursday if Friday is a holiday)."""
    us = p.index[p["us_open"]]
    us = us[(us >= start) & (us <= end)]
    s = pd.Series(us, index=us)
    wk = s.groupby(s.index.to_period("W-FRI")).max()
    return pd.DatetimeIndex(wk.values)


def next_us_day(p, d):
    us = p.index[p["us_open"] & (p.index > d)]
    return us[0] if len(us) else None


def simulate(R, target_fn, p, start, end, exec_lag=1, cash="krw", band=0.0,
             brake=None, brake_check="daily", base_nav=None, fx=None, irx=None):
    """
    R: DataFrame of daily simple KRW returns for the risky assets (calendar grid).
    target_fn(date) -> array of target weights (sum<=1) using data through `date` only.
    brake: None, or dict(kind='mtd', L=0.08, cut=0.5)  -> if month-to-date KRW loss <= -L, scale to cut until month end
                     or dict(kind='dd', D=0.15, cut=0.5, recover_frac=0.5) -> uses base_nav (un-braked strategy NAV):
                        drawdown <= -D => cut; restore when drawdown >= -D*recover_frac.
    Trades from signals/brakes execute at the close of the next US trading day (exec_lag=1).
    """
    R = R.loc[start:end].fillna(0.0)
    idx = R.index
    na = R.shape[1]
    sig = set(signal_dates(p, start, end))
    us_open = p["us_open"].reindex(idx).fillna(False).values.astype(bool)
    us_days = idx[us_open]
    pos = {d: k for k, d in enumerate(us_days)}

    def nxt(d):
        k = us_days.searchsorted(d, side="right")
        return us_days[k] if k < len(us_days) else None

    if cash == "usd":
        fxr = fx.pct_change(fill_method=None).reindex(idx).fillna(0).values
        rf = (irx.reindex(idx).ffill() / 100 / 365).fillna(0).values
    if brake is not None and brake["kind"] == "dd":
        bn = base_nav.reindex(idx).ffill()
        bpk = bn.cummax()
        bdd = (bn / bpk - 1).values
    Rv = R.values
    nav = np.empty(len(idx)); W = np.zeros((len(idx), na))
    h = np.zeros(na); c = 1.0
    tgt_base = np.zeros(na); mult = 1.0
    turnover = 0.0; n_trades = 0
    queue = {}  # date -> list of events
    br_on = False; br_log = []
    month_start_nav = 1.0

    def push(d, ev):
        if d is not None:
            queue.setdefault(d, []).append(ev)

    def trade(V, h, c, only_rebal):
        nonlocal turnover, n_trades
        tgt = tgt_base * mult
        cur = h / V
        dw = np.abs(tgt - cur).sum()
        if only_rebal and band > 0 and dw < band:
            return h, c
        if dw > 1e-9:
            turnover += dw; n_trades += 1
        h = tgt * V
        return h, V - h.sum()

    for t, d in enumerate(idx):
        if t > 0:
            h = h * (1 + Rv[t])
            if cash == "usd":
                c *= (1 + fxr[t]) * (1 + rf[t])
        V = h.sum() + c
        if t > 0 and d.month != idx[t - 1].month:
            month_start_nav = nav[t - 1]
            if brake is not None and brake["kind"] == "mtd" and br_on:
                br_on = False
                push(d if us_open[t] else nxt(d), ("mult", 1.0))
                br_log.append((d, "off"))
        evs = queue.pop(d, [])
        if t == 0:
            evs = [("rebal", np.asarray(target_fn(d), float))]
        if evs:
            only_rebal = all(e[0] == "rebal" for e in evs) and t > 0
            for e in evs:
                if e[0] == "rebal":
                    tgt_base = e[1]
                else:
                    mult = e[1]
            h, c = trade(V, h, c, only_rebal)
        V = h.sum() + c
        if d in sig and t > 0:
            w = np.asarray(target_fn(d), float)
            if exec_lag:
                push(nxt(d), ("rebal", w))
            else:
                tgt_base = w
                h, c = trade(V, h, c, True)
        nav[t] = V
        W[t] = h / V
        if brake is not None and us_open[t] and (brake_check == "daily" or d in sig):
            if brake["kind"] == "mtd" and not br_on and V / month_start_nav - 1 <= -brake["L"]:
                ex = nxt(d)
                if ex is not None and ex.month == d.month:
                    br_on = True; push(ex, ("mult", brake.get("cut", 0.5))); br_log.append((d, "on"))
            elif brake["kind"] == "dd":
                if not br_on and bdd[t] <= -brake["D"]:
                    br_on = True; push(nxt(d), ("mult", brake.get("cut", 0.5))); br_log.append((d, "on"))
                elif br_on and bdd[t] >= -brake["D"] * brake.get("recover_frac", 0.5):
                    br_on = False; push(nxt(d), ("mult", 1.0)); br_log.append((d, "off"))
    nav = pd.Series(nav, index=idx)
    W = pd.DataFrame(W, index=idx, columns=R.columns)
    yrs = (idx[-1] - idx[0]).days / 365.25
    return {"nav": nav, "w": W, "turnover": turnover / yrs, "trades_per_yr": n_trades / yrs, "brake_log": br_log}


def summarize(res, start_eval=None, end_eval=None):
    nav = res["nav"]
    if start_eval is not None:
        nav = nav.loc[:end_eval]
    mr = monthly(nav)
    if start_eval is not None:
        mr = mr.loc[start_eval:end_eval]
    t = tails(mr, nav.loc[start_eval:] if start_eval else nav)
    t["avg_expo"] = res["w"].sum(axis=1).loc[start_eval:end_eval].mean() if start_eval else res["w"].sum(axis=1).mean()
    t["turnover"] = res["turnover"]
    t["trades_yr"] = res["trades_per_yr"]
    return t
