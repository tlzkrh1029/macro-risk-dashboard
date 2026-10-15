#!/usr/bin/env python3
"""Empirical test of the v2 sizing rule (docs/v2-design.md, '결론 규칙: 사이징표').

What it does
  1. Rebuilds the six v2 traffic lights daily from cached data (2007-05 .. 2026-10):
       layer-1 diagnosis (GSW/ACM 4-week changes + S&P 500 4-week return, weekly by default),
       credit, rate vol, equity vol, hedge, funding/dollar; plus composite alarms A and B
       (rolling 5-business-day changes, valid for 10 business days after the last firing).
  2. Maps state -> exposure (공격/중립/방어 = 100/70/40 and 100/60/20), one-step-up rule,
     signals lagged one business day, applied to ^GSPC (price index) and BTC-USD.
  3. Baselines: buy-and-hold, 200-day MA filter (100/40), vol target (min(1, 15%/rv20)),
     composite-alarm-only (40% while A or B valid), constant exposure at v2's average,
     and leave-one-bucket-out ablations.
  4. Metrics, episodes, structural-red statistics, forward-risk-by-state, placebo and bootstrap.

Proxies (FRED unreachable): credit = HYG/IEI 20-business-day log change (HY OAS unavailable),
funding/dollar = DXY 20-day change only (no NFCI, no SOFR-IORB, no broad dollar).
Everything else follows the v2 text literally; interpretation choices are listed in CFG below.

Usage: python3 -I backtest_v2_sizing.py   (repo is read only; no bytecode is written into it)
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)

REPO = Path("/home/user/macro-risk-dashboard")
SCRATCH = Path("/tmp/claude-0/-home-user-macro-risk-dashboard/157fe875-3352-5fc5-b8c1-31e705d9a652/scratchpad")
OUT = Path(__file__).resolve().parent
EXTRA_YAHOO = SCRATCH / "data" / "yahoo.csv"
SPX_END = "2026-10-13"  # last real ^GSPC close in the cache (10-14 is a forward-filled row)
RNG = np.random.default_rng(20261014)

STATE_NAMES = {0: "공격", 1: "중립", 2: "방어"}
BUCKETS = ["L1_diag", "credit", "rate_vol", "eq_vol", "hedge", "funding"]

CFG = dict(
    # layer-1 thresholds (v2 table; nominal ±20bp, BE ±15bp, S&P ±2%)
    nom_th=20.0, be_th=15.0, spx_th=0.02,
    l1_cadence="weekly",          # v2: '주 1회, 주말'. 'daily' = rolling 20 business days
    l1_fall_flat="green",         # falling nominal, expected-path & real driven, S&P '→': v2 leaves this cell undefined
    real_pct_window=756, real_pct_q=0.95,  # 3y rolling 95th pct of 10y real (GSW TIPSY10), trailing data only
    # credit proxy: HYG/IEI 20d log change thresholds standing in for HY OAS +15% / +30%
    credit_yellow=-0.015, credit_red=-0.030,
    credit_level="none",          # 'none' | 'dd' (HYG/IEI 252d drawdown <= credit_dd) | 'calendar' (approx HY OAS>450 windows)
    credit_dd=-0.05,
    # rate vol (literal)
    move5_y=0.20, tyx20_y=40.0, tyx5_r=25.0, move_lvl_r=126.0,
    # equity vol (literal)
    vixr_y=0.95, vixr_r=1.00,
    # hedge (literal)
    corr_win=63, corr_y=0.0, corr_r=0.3,
    # funding/dollar proxy: DXY 20d change only; red side unavailable (NFCI, SOFR-IORB)
    dxy20_y=0.03,
    # composite alarms (5 business-day log changes, as in analysis/verify_v1.py)
    comp_valid_days=10,
    # sizing
    expo=(1.0, 0.7, 0.4),
    cadence="daily",              # 'daily' or 'weekly' (weekend conclusion; composite firing can still force 방어 any day)
    lag=1,
    use_composite=True,
    drop=(),                      # buckets removed for ablations
)

# Approximate windows when ICE BofA HY OAS (BAMLH0A0HYM2) was above 450bp, written from public
# record / memory because FRED is blocked here. NOT verified in this environment; used only in an
# explicitly labelled illustrative scenario.
HY_OAS_GT450_APPROX = [
    ("2007-11-15", "2013-06-30"), ("2014-12-10", "2015-01-31"), ("2015-08-20", "2016-09-30"),
    ("2018-12-10", "2019-01-15"), ("2020-03-02", "2020-11-10"), ("2022-06-13", "2022-07-29"),
    ("2022-09-20", "2022-11-15"), ("2023-03-13", "2023-03-24"), ("2025-04-04", "2025-04-22"),
]


# ----------------------------------------------------------------------------- data
def import_verify():
    spec = importlib.util.spec_from_file_location("verify_v1", REPO / "analysis" / "verify_v1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.CACHE = REPO / "analysis" / ".cache"
    return mod


def build_inputs():
    v = import_verify()
    nom, tips, acm, y = v.load()
    idx = y.index
    raw = pd.read_csv(v.CACHE / "yahoo.csv", index_col=0, parse_dates=True)  # not forward filled, has weekends
    ex = pd.read_csv(EXTRA_YAHOO, index_col=0, parse_dates=True)
    ex = ex[ex.index.dayofweek < 5].ffill(limit=3).reindex(idx)
    rates = pd.DataFrame({
        "nom": nom["SVENY10"], "real": tips["TIPSY10"], "be": tips["BKEVEN10"],
        "tp": acm["ACMTP10"], "rn": acm["ACMRNY10"],
    })
    rates = rates[rates.index.dayofweek < 5].reindex(idx).ffill(limit=3)
    return dict(y=y, idx=idx, raw=raw, ex=ex, rates=rates)


# ----------------------------------------------------------------------------- layer 1
def classify_l1(dn, dr, dbe, dtp, drn, s4, cfg):
    """Vectorised v2 layer-1 table. Inputs in bp (rates) and fraction (S&P). Returns (label, colour 0/1)."""
    nt, bt, st = cfg["nom_th"], cfg["be_th"], cfg["spx_th"]
    flat = dn.abs() < nt
    up, dn_ = dn >= nt, dn <= -nt
    d1_rn = drn.abs() > dtp.abs()
    d2_real = dr.abs() > dbe.abs()
    sp_up, sp_dn = s4 >= st, s4 <= -st
    fall_flat_col = 0 if cfg["l1_fall_flat"] == "green" else 1
    conds = [
        flat & (dbe.abs() < bt),
        flat & (dbe >= bt),
        flat & (dbe <= -bt),
        up & d1_rn & d2_real & sp_up,
        up & d1_rn & d2_real & sp_dn,
        up & d1_rn & d2_real,
        up & d1_rn & ~d2_real,
        up & ~d1_rn & d2_real,
        up & ~d1_rn & ~d2_real,
        dn_ & d1_rn & d2_real & sp_up,
        dn_ & d1_rn & d2_real & sp_dn,
        dn_ & d1_rn & d2_real,
        dn_ & d1_rn & ~d2_real,
        dn_ & ~d1_rn & d2_real,
        dn_ & ~d1_rn & ~d2_real,
    ]
    labels = ["무판정", "완화+리플레이션", "디스인플레+실질긴축",
              "성장주도 재가격", "긴축재가격-정책충격", "긴축재가격-경계구간", "인플레 대응 긴축",
              "재정·공급·불확실성 TP", "인플레 불확실성 TP",
              "완화 랠리", "성장둔화·침체우려", "완화기대(S&P→, 미정의)", "디스인플레 완화",
              "안전자산 수요", "인플레 위험 축소"]
    colours = [0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, fall_flat_col, 0, 1, 0]
    valid = pd.concat([dn, dr, dbe, dtp, drn, s4], axis=1).notna().all(axis=1)
    lab = pd.Series(np.select(conds, labels, default="NA"), index=dn.index).where(valid)
    col = pd.Series(np.select(conds, colours, default=np.nan), index=dn.index).where(valid)
    return lab, col


def layer1(inp, cfg):
    r, idx = inp["rates"], inp["idx"]
    spx = inp["y"]["^GSPC"]
    real_q = r["real"].rolling(cfg["real_pct_window"], min_periods=500).quantile(cfg["real_pct_q"])
    real_hi = r["real"] >= real_q
    if cfg["l1_cadence"] == "weekly":
        fri = idx[idx.dayofweek == 4]
        rw = r.reindex(fri)
        d = rw.diff(4) * 100
        s4 = spx.reindex(fri).pct_change(4, fill_method=None)
        hi = real_hi.reindex(fri)
    else:
        d = r.diff(20) * 100
        s4 = spx.pct_change(20, fill_method=None)
        hi = real_hi
    lab, col = classify_l1(d["nom"], d["real"], d["be"], d["tp"], d["rn"], s4, cfg)
    col = col.where(~((col == 1) & hi.fillna(False)), 2)
    out = pd.DataFrame({"label": lab, "colour": col, "dnom": d["nom"], "dreal": d["real"], "dbe": d["be"],
                        "dtp": d["tp"], "drn": d["rn"], "spx4": s4})
    if cfg["l1_cadence"] == "weekly":
        out = out.reindex(idx).ffill(limit=5)
    return out


# ----------------------------------------------------------------------------- layer 2
def layer2(inp, cfg):
    y, idx = inp["y"], inp["idx"]
    L = np.log(y)
    cr20 = (L["HYG"] - L["IEI"]).diff(20)
    ratio = y["HYG"] / y["IEI"]
    cr_dd = ratio / ratio.rolling(252, min_periods=120).max() - 1
    credit_red = cr20 <= cfg["credit_red"]
    if cfg["credit_level"] == "dd":
        credit_red = credit_red | (cr_dd <= cfg["credit_dd"])
    elif cfg["credit_level"] == "calendar":
        cal = pd.Series(False, index=idx)
        for a, b in HY_OAS_GT450_APPROX:
            cal.loc[a:b] = True
        credit_red = credit_red | cal
    credit_y = cr20 <= cfg["credit_yellow"]

    move = y["^MOVE"]
    move5 = move.pct_change(5, fill_method=None)
    tyx5 = y["^TYX"].diff(5) * 100
    tyx20 = y["^TYX"].diff(20) * 100
    rv_y = (move5 >= cfg["move5_y"]) | (tyx20 >= cfg["tyx20_y"])
    rv_r = ((tyx5 >= cfg["tyx5_r"]) & (move5 >= cfg["move5_y"])) | (move > cfg["move_lvl_r"])

    vixr = y["^VIX"] / y["^VIX3M"]
    ev_y, ev_r = vixr > cfg["vixr_y"], vixr > cfg["vixr_r"]

    lr = np.log(y[["^GSPC", "TLT"]]).diff()
    corr = lr["^GSPC"].rolling(cfg["corr_win"]).corr(lr["TLT"])
    hd_y, hd_r = corr > cfg["corr_y"], corr > cfg["corr_r"]

    dxy20 = y["DX-Y.NYB"].pct_change(20, fill_method=None)
    fd_y = dxy20 >= cfg["dxy20_y"]
    fd_r = pd.Series(False, index=idx)  # NFCI > 0 and SOFR-IORB unavailable

    def colour(yel, red, *inputs):
        c = pd.Series(np.where(red, 2, np.where(yel, 1, 0)), index=idx, dtype=float)
        ok = pd.concat(inputs, axis=1).notna().all(axis=1)
        return c.where(ok)

    cols = pd.DataFrame({
        "credit": colour(credit_y, credit_red, cr20),
        "rate_vol": colour(rv_y, rv_r, move5, tyx20),
        "eq_vol": colour(ev_y, ev_r, vixr),
        "hedge": colour(hd_y, hd_r, corr),
        "funding": colour(fd_y, fd_r, dxy20),
    })
    raw = pd.DataFrame({"hyg_iei_20d": cr20, "hyg_iei_dd252": cr_dd, "move": move, "move5": move5, "tyx5_bp": tyx5,
                        "tyx20_bp": tyx20, "vix_vix3m": vixr, "spx_tlt_corr63": corr, "dxy20": dxy20})
    return cols, raw


def composites(inp, cfg):
    y = inp["y"]
    L = np.log(y)
    c5 = (L["HYG"] - L["IEI"]).diff(5)
    m5, t5, d5, v5 = L["^MOVE"].diff(5), L["TLT"].diff(5), L["DX-Y.NYB"].diff(5), L["^VIX"].diff(5)
    A = ((c5 < -0.005) & (m5 > 0.10) & (t5 < -0.01) & (d5 > 0.005)).fillna(False)
    B = ((c5 < -0.005) & (m5 > 0.10) & (t5 > 0.01) & (v5 > 0.15)).fillna(False)
    w = cfg["comp_valid_days"] + 1  # firing day + 10 business days (v2 example: fired 10/2, valid through 10/16)
    vA = A.astype(float).rolling(w, min_periods=1).max().astype(bool)
    vB = B.astype(float).rolling(w, min_periods=1).max().astype(bool)
    return pd.DataFrame({"A": A, "B": B, "validA": vA, "validB": vB, "valid": vA | vB, "fire": A | B})


# ----------------------------------------------------------------------------- state machine
def all_colours(inp, cfg):
    l1 = layer1(inp, cfg)
    l2, raw2 = layer2(inp, cfg)
    cols = pd.concat([l1["colour"].rename("L1_diag"), l2], axis=1)[BUCKETS]
    return cols, l1, raw2


def raw_state(cols, comp, cfg):
    use = [b for b in BUCKETS if b not in cfg["drop"]]
    c = cols[use]
    ok = c.notna().all(axis=1)
    reds = (c == 2).sum(axis=1)
    yels = (c == 1).sum(axis=1)
    cv = comp["valid"] if cfg["use_composite"] else pd.Series(False, index=cols.index)
    st = np.where((reds >= 2) | cv, 2, np.where((reds == 1) | (yels >= 2), 1, 0))
    st = pd.Series(st, index=cols.index, dtype=float).where(ok)
    return st, reds.where(ok), yels.where(ok)


def apply_one_step(raw, comp, cfg):
    """One-step-up rule. 'daily': evaluated every business day. 'weekly': conclusion on Fridays only,
    but a composite firing (A or B) on any day moves the state to 방어 immediately."""
    r = raw.dropna()
    out = np.empty(len(r))
    fire = comp["fire"].reindex(r.index).values if cfg["use_composite"] else np.zeros(len(r), bool)
    is_fri = r.index.dayofweek == 4
    cur = r.iloc[0]
    vals = r.values
    for i in range(len(r)):
        if cfg["cadence"] == "daily" or is_fri[i]:
            tgt = vals[i]
            cur = tgt if tgt >= cur else cur - 1
        elif fire[i]:
            cur = 2.0
        out[i] = cur
    return pd.Series(out, index=r.index)


def v2_state(cols, comp, cfg):
    raw, reds, yels = raw_state(cols, comp, cfg)
    st = apply_one_step(raw, comp, cfg)
    return st, raw, reds, yels


def to_expo(state, mapping):
    return state.map({0.0: mapping[0], 1.0: mapping[1], 2.0: mapping[2]})


# ----------------------------------------------------------------------------- baselines
def ma_filter(px_native, idx, n=200, hi=1.0, lo=0.4):
    s = px_native.dropna()
    sig = (s > s.rolling(n).mean()).where(s.rolling(n).mean().notna())
    e = sig.map({True: hi, False: lo})
    return e.reindex(idx, method="ffill")


def vol_target(px_native, idx, ann, target=0.15, n=20):
    s = px_native.dropna()
    rv = s.pct_change(fill_method=None).rolling(n).std() * np.sqrt(ann)
    e = np.minimum(1.0, target / rv)
    return e.reindex(idx, method="ffill")


# ----------------------------------------------------------------------------- backtest and metrics
def strat_returns(r, expo, lag=1, cash=None):
    e = expo.reindex(r.index).shift(lag)
    out = e * r
    if cash is not None:
        out = out + (1 - e) * cash.reindex(r.index).fillna(0)
    return out, e


def metrics(sr, e, start, end):
    sr = sr.loc[start:end].dropna()
    e = e.loc[sr.index]
    yrs = (sr.index[-1] - sr.index[0]).days / 365.25
    ppy = len(sr) / yrs
    eq = (1 + sr).cumprod()
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = sr.std() * np.sqrt(ppy)
    sharpe = sr.mean() * ppy / vol if vol > 0 else np.nan
    mdd = (eq / eq.cummax() - 1).min()
    return dict(CAGR=cagr, Vol=vol, Sharpe=sharpe, MaxDD=mdd, Calmar=cagr / abs(mdd) if mdd < 0 else np.nan,
                AvgExpo=e.mean(), CAGR_per_expo=cagr / e.mean(), Turnover_yr=e.diff().abs().sum() / yrs)


def fmt_metrics(rows):
    df = pd.DataFrame(rows).T
    f = df.copy()
    for c in ["CAGR", "Vol", "MaxDD", "AvgExpo", "CAGR_per_expo"]:
        f[c] = df[c].map(lambda x: f"{x:+.1%}" if c in ("CAGR", "MaxDD") else f"{x:.1%}")
    for c in ["Sharpe", "Calmar", "Turnover_yr"]:
        f[c] = df[c].map(lambda x: f"{x:.2f}")
    return f


def state_stats(state, start, end):
    s = state.loc[start:end].dropna()
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    share = s.value_counts(normalize=True).reindex([0.0, 1.0, 2.0]).fillna(0)
    ch = (s.diff().fillna(0) != 0).sum() / yrs
    return {f"%{STATE_NAMES[k]}": f"{v:.0%}" for k, v in share.items()} | {"changes/yr": f"{ch:.1f}"}


def runs(mask):
    """Return list of (start, end, length) for consecutive True runs."""
    m = mask.fillna(False).astype(int)
    g = (m.diff() != 0).cumsum()
    out = []
    for _, seg in m.groupby(g):
        if seg.iloc[0] == 1:
            out.append((seg.index[0], seg.index[-1], len(seg)))
    return out


def stationary_bootstrap_idx(n, mean_block, rng):
    p = 1.0 / mean_block
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    jumps = rng.random(n) < p
    starts = rng.integers(n, size=n)
    for i in range(1, n):
        idx[i] = starts[i] if jumps[i] else (idx[i - 1] + 1) % n
    return idx


def ann_sharpe(x, ppy):
    sd = x.std()
    return x.mean() / sd * np.sqrt(ppy) if sd > 0 else np.nan


# ----------------------------------------------------------------------------- main
def main():
    inp = build_inputs()
    y, idx, raw, ex = inp["y"], inp["idx"], inp["raw"], inp["ex"]
    cfg = dict(CFG)
    cols, l1, raw2 = all_colours(inp, cfg)
    comp = composites(inp, cfg)
    state, rawst, reds, yels = v2_state(cols, comp, cfg)
    start = state.index[0]
    print(f"Signals available from {start.date()} (first date all six buckets are defined); "
          f"GSW/ACM through {inp['rates'].dropna().drop_duplicates().index[-1].date()} (ffilled 3d), "
          f"Yahoo through {idx[-1].date()}")

    # --- returns
    spx_r = y["^GSPC"].pct_change(fill_method=None).loc[:SPX_END]
    btc_r = y["BTC-USD"].pct_change(fill_method=None)
    spy_r = ex["SPY"].pct_change(fill_method=None).loc[:SPX_END]
    cash = (ex["^IRX"] / 100 / 261).shift(1)

    spx_raw = raw["^GSPC"].dropna()
    btc_raw = raw["BTC-USD"].dropna()
    ma_spx = ma_filter(spx_raw, idx)
    vt_spx = vol_target(spx_raw, idx, 252)
    ma_btc = ma_filter(btc_raw, idx)
    vt_btc = vol_target(btc_raw, idx, 365)
    vt_btc60 = vol_target(btc_raw, idx, 365, target=0.60)

    def run(cfg_over=None, asset="spx", mapping=None, cash_on=False):
        c = dict(CFG)
        c.update(cfg_over or {})
        cl, _, _ = all_colours(inp, c)
        cp = composites(inp, c)
        st, rs, rd, yl = v2_state(cl, cp, c)
        e = to_expo(st, mapping or c["expo"])
        r = {"spx": spx_r, "btc": btc_r, "spy": spy_r}[asset]
        sr, eu = strat_returns(r, e, c["lag"], cash if cash_on else None)
        return sr, eu, st

    # common evaluation windows
    s0 = max(start, ma_spx.dropna().index[0], vt_spx.dropna().index[0]) + pd.offsets.BDay(2)
    s_end = pd.Timestamp(SPX_END)
    b0 = max(start, ma_btc.dropna().index[0], vt_btc.dropna().index[0]) + pd.offsets.BDay(2)
    b_end = idx[-1]

    # ===================================================================== 1. traffic lights
    print("\n=== 1. Bucket traffic lights: share of days (from %s) ===" % s0.date())
    cc = cols.loc[s0:s_end]
    t = pd.DataFrame({b: cc[b].value_counts(normalize=True).reindex([0.0, 1.0, 2.0]).fillna(0) for b in BUCKETS}).T
    t.columns = ["green", "yellow", "red"]
    print((t * 100).round(1).to_string())
    print("\nLongest red streaks per bucket (business days) and red share 2022-2026:")
    for b in BUCKETS:
        rr = sorted(runs(cc[b] == 2), key=lambda x: -x[2])[:3]
        s = ", ".join(f"{a.date()}..{e.date()} ({n}d)" for a, e, n in rr) or "none"
        print(f"  {b:9s} red 2022+: {(cc[b].loc['2022':] == 2).mean():5.1%} | top runs: {s}")
    print("\nRed share by year (%):")
    ry = (cc == 2).groupby(cc.index.year).mean().mul(100).round(0)
    ry["reds>=2"] = (reds.loc[s0:s_end] >= 2).groupby(reds.loc[s0:s_end].index.year).mean().mul(100).round(0)
    ry["comp_valid"] = comp["valid"].loc[s0:s_end].groupby(comp.loc[s0:s_end].index.year).mean().mul(100).round(0)
    ry["%방어(final)"] = (state.loc[s0:s_end] == 2).groupby(state.loc[s0:s_end].index.year).mean().mul(100).round(0)
    print(ry.astype(int).to_string())
    print("\nLayer-1 diagnosis label frequency (weekly obs):")
    l1w = l1.loc[s0:s_end]
    l1w = l1w[l1w.index.dayofweek == 4]
    print((l1w["label"].value_counts(normalize=True) * 100).round(1).to_string())
    why = pd.DataFrame({"reds2": reds >= 2, "comp": comp["valid"]}).loc[s0:s_end][rawst.loc[s0:s_end] == 2]
    print(f"\nRaw 방어 days: {len(why)}; caused by reds>=2 only {(why.reds2 & ~why.comp).mean():.0%}, "
          f"composite only {(~why.reds2 & why.comp).mean():.0%}, both {(why.reds2 & why.comp).mean():.0%}")
    pairs = cc[(cc == 2).sum(axis=1) >= 2].apply(lambda r: "+".join(b for b in BUCKETS if r[b] == 2), axis=1)
    print("Most common red combinations on reds>=2 days:")
    print(pairs.value_counts(normalize=True).head(6).mul(100).round(1).to_string())
    print("\nStructural vs transient red: share of each bucket's red days that sit in runs of >=20 / >=63 business days")
    for b in BUCKETS:
        rr = runs(cc[b] == 2)
        tot = sum(n for _, _, n in rr)
        if tot:
            print(f"  {b:9s} red days {tot:5d}: in runs>=20d {sum(n for *_, n in rr if n >= 20) / tot:4.0%}, "
                  f">=63d {sum(n for *_, n in rr if n >= 63) / tot:4.0%}; number of red spells {len(rr)}")
    mv_lvl = (raw2["move"] > CFG["move_lvl_r"]).loc[s0:s_end]
    print(f"  rate_vol red days explained by MOVE>126 level alone: {(mv_lvl & (cc['rate_vol'] == 2)).sum() / (cc['rate_vol'] == 2).sum():.0%}")
    both = ((cc["L1_diag"] == 2) & (cc["hedge"] == 2))
    print(f"  L1_diag and hedge both red: 2026 {both.loc['2026'].mean():.0%} of days, since 2026-09-01 {both.loc['2026-09-01':].mean():.0%}; "
          f"hedge red since 2026-05-01 {(cc['hedge'].loc['2026-05-01':] == 2).mean():.0%}")
    print(f"Composite alarm firings (days) since {s0.date()}: A {int(comp['A'].loc[s0:].sum())}, "
          f"B {int(comp['B'].loc[s0:].sum())}; share of days valid: A {comp['validA'].loc[s0:].mean():.1%}, "
          f"B {comp['validB'].loc[s0:].mean():.1%}")

    # ===================================================================== 2. current readout
    print("\n=== 2. Current readout vs the v2 worked example (10/13-14) ===")
    snap = pd.concat([cols, l1[["label", "dnom", "dreal", "dbe", "dtp", "drn", "spx4"]], raw2,
                      comp[["A", "B", "valid"]], rawst.rename("raw_state"), state.rename("state")], axis=1)
    print(snap.loc["2026-09-21":"2026-10-14"].round(3).to_string())
    a_days = comp.index[comp["A"] & (comp.index >= "2026-09-01")]
    print("A firing days Sep-Oct 2026:", [str(d.date()) for d in a_days])

    # ===================================================================== 3. S&P performance
    print(f"\n=== 3. S&P 500 (^GSPC price index, no dividends; cash 0%; rf=0 Sharpe) {s0.date()} .. {s_end.date()} ===")
    strategies = {}
    one = pd.Series(1.0, index=idx)
    strategies["Buy&hold"] = strat_returns(spx_r, one, 0)
    sr, eu, st_base = run()
    strategies["v2 100/70/40"] = (sr, eu)
    sr2, eu2, _ = run(mapping=(1.0, 0.6, 0.2))
    strategies["v2 100/60/20"] = (sr2, eu2)
    strategies["MA200 100/40"] = strat_returns(spx_r, ma_spx, 1)
    strategies["VolTarget 15%"] = strat_returns(spx_r, vt_spx, 1)
    ce = to_expo(pd.Series(np.where(comp["valid"], 2.0, 0.0), index=idx), (1.0, 0.7, 0.4))
    strategies["Composite-only 100/40"] = strat_returns(spx_r, ce, 1)
    avg_e = eu.loc[s0:s_end].mean()
    strategies[f"Constant {avg_e:.0%}"] = strat_returns(spx_r, pd.Series(avg_e, index=idx), 0)
    vtc = np.minimum(vt_spx, pd.Series(np.where(comp["valid"], 0.4, 1.0), index=idx))
    strategies["VolTarget 15% + composite cap 40%"] = strat_returns(spx_r, vtc, 1)
    rows = {k: metrics(v[0], v[1], s0, s_end) for k, v in strategies.items()}
    print(fmt_metrics(rows).to_string())
    print("\nCost drag: CAGR net of a one-way cost per unit of exposure change (turnover x cost, approx.)")
    cd = {}
    for k, m in rows.items():
        cd[k] = {"gross": f"{m['CAGR']:+.1%}", "10bp": f"{m['CAGR'] - m['Turnover_yr'] * 0.0010:+.1%}",
                 "30bp": f"{m['CAGR'] - m['Turnover_yr'] * 0.0030:+.1%}", "turnover/yr": f"{m['Turnover_yr']:.1f}"}
    print(pd.DataFrame(cd).T.to_string())
    ev2 = to_expo(st_base, CFG["expo"]).loc[s0:s_end]
    print("Spearman corr of daily exposure paths: v2 vs VolTarget %.2f, v2 vs MA200 %.2f, v2 vs Composite-only %.2f" % (
        ev2.corr(vt_spx.loc[s0:s_end], method="spearman"), ev2.corr(ma_spx.loc[s0:s_end], method="spearman"),
        ev2.corr(ce.loc[s0:s_end], method="spearman")))
    print("\nState statistics (v2 base, final state after one-step-up):", state_stats(st_base, s0, s_end))
    print("Raw state (before one-step-up):", state_stats(rawst, s0, s_end))
    ma_state = (ma_spx < 1).astype(float) * 2
    print("MA200 below-share and switches/yr:", state_stats(ma_state, s0, s_end))
    print("Composite-only state:", state_stats(pd.Series(np.where(comp['valid'], 2.0, 0.0), index=idx), s0, s_end))

    # ablations
    print("\n--- Ablations (v2 100/70/40, remove one input) ---")
    abl = {"v2 base": rows["v2 100/70/40"]}
    st_abl = {}
    for b in BUCKETS:
        sr_, eu_, st_ = run({"drop": (b,)})
        abl[f"- {b}"] = metrics(sr_, eu_, s0, s_end)
        st_abl[f"- {b}"] = state_stats(st_, s0, s_end)
    sr_, eu_, st_ = run({"use_composite": False})
    abl["- composite"] = metrics(sr_, eu_, s0, s_end)
    st_abl["- composite"] = state_stats(st_, s0, s_end)
    sr_, eu_, st_ = run({"drop": ("L1_diag", "hedge")})
    abl["- L1_diag - hedge"] = metrics(sr_, eu_, s0, s_end)
    st_abl["- L1_diag - hedge"] = state_stats(st_, s0, s_end)
    fa = fmt_metrics(abl)
    fa = fa.join(pd.DataFrame(st_abl | {"v2 base": state_stats(st_base, s0, s_end)}).T)
    print(fa.to_string())

    # sensitivity
    print("\n--- Sensitivity (one change at a time vs base) ---")
    sens_specs = {
        "base": ({}, "spx", None, False),
        "credit map -0.5/-1.0 (2026-anchored)": ({"credit_yellow": -0.005, "credit_red": -0.010}, "spx", None, False),
        "credit map -1.0/-2.0": ({"credit_yellow": -0.010, "credit_red": -0.020}, "spx", None, False),
        "credit map -2.0/-4.0": ({"credit_yellow": -0.020, "credit_red": -0.040}, "spx", None, False),
        "credit map -4.0/-7.5 (freq-matched, assumed)": ({"credit_yellow": -0.040, "credit_red": -0.075}, "spx", None, False),
        "credit + level proxy (HYG/IEI dd252<=-5%)": ({"credit_level": "dd"}, "spx", None, False),
        "credit + approx HY OAS>450 calendar": ({"credit_level": "calendar"}, "spx", None, False),
        "funding DXY 20d +4% (vol-adjusted)": ({"dxy20_y": 0.04}, "spx", None, False),
        "L1 daily rolling 20d": ({"l1_cadence": "daily"}, "spx", None, False),
        "L1 fall/RN+real/S&P-> = yellow": ({"l1_fall_flat": "yellow"}, "spx", None, False),
        "weekly conclusion (Fri) + daily composite": ({"cadence": "weekly"}, "spx", None, False),
        "lag 2 days": ({"lag": 2}, "spx", None, False),
        "exposure 100/60/20": ({}, "spx", (1.0, 0.6, 0.2), False),
        "cash earns T-bill (^IRX)": ({}, "spx", None, True),
        "SPY total return": ({}, "spy", None, False),
    }
    sens, sstat = {}, {}
    for k, (over, asset, mp, cash_on) in sens_specs.items():
        sr_, eu_, st_ = run(over, asset, mp, cash_on)
        sens[k] = metrics(sr_, eu_, s0, s_end)
        sstat[k] = state_stats(st_, s0, s_end)
    bh_cash = metrics(*strat_returns(spx_r, one, 0), s0, s_end)
    spy_bh = metrics(*strat_returns(spy_r, one, 0), s0, s_end)
    sens["(ref) B&H ^GSPC"] = bh_cash
    sens["(ref) B&H SPY TR"] = spy_bh
    ma_cash = metrics(*strat_returns(spx_r, ma_spx, 1, cash), s0, s_end)
    sens["(ref) MA200 with T-bill cash"] = ma_cash
    print(fmt_metrics(sens).join(pd.DataFrame(sstat).T).to_string())

    print("\nCalendar-year returns (^GSPC price):")
    yr = {}
    for k in ["Buy&hold", "v2 100/70/40", "MA200 100/40", "VolTarget 15%", "Composite-only 100/40"]:
        w = strategies[k][0].loc[s0:s_end].fillna(0)
        yr[k] = (1 + w).groupby(w.index.year).prod() - 1
    yr = pd.DataFrame(yr)
    yr["v2 avgExpo"] = to_expo(st_base, CFG["expo"]).loc[s0:s_end].groupby(lambda d: d.year).mean()
    yr["v2 - B&H"] = yr["v2 100/70/40"] - yr["Buy&hold"]
    print(yr.map(lambda x: f"{x:+.1%}").to_string())

    # sub-periods
    print("\n--- Sub-periods ---")
    for a, b in [(s0, pd.Timestamp("2016-12-30")), (pd.Timestamp("2017-01-03"), s_end), (pd.Timestamp("2022-01-03"), s_end)]:
        sub = {k: metrics(strategies[k][0], strategies[k][1], a, b)
               for k in ["Buy&hold", "v2 100/70/40", "v2 100/60/20", "MA200 100/40", "VolTarget 15%", "Composite-only 100/40"]}
        print(f"{a.date()} .. {b.date()}")
        print(fmt_metrics(sub)[["CAGR", "Vol", "Sharpe", "MaxDD", "Calmar", "AvgExpo"]].to_string())

    # ===================================================================== 4. episodes
    print("\n=== 4. Episodes (^GSPC). ret = total return over window, mdd = max drawdown inside window ===")
    episodes = {
        "2008 GFC": ("2007-10-09", "2009-12-31"), "2011 US downgrade": ("2011-07-01", "2012-03-30"),
        "2015-16": ("2015-07-01", "2016-06-30"), "2018Q4": ("2018-09-20", "2019-04-30"),
        "2020 Covid": ("2020-02-19", "2020-08-31"), "2022": ("2022-01-03", "2022-12-30"),
        "2023": ("2023-01-03", "2023-12-29"), "2025-04 tariffs": ("2025-02-19", "2025-06-30"),
        "2026-09/10": ("2026-08-19", SPX_END),
    }
    ep_keys = ["Buy&hold", "v2 100/70/40", "v2 100/60/20", "MA200 100/40", "VolTarget 15%", "Composite-only 100/40"]
    rows = []
    for name, (a, b) in episodes.items():
        px = y["^GSPC"].loc[a:b]
        pk = px.loc[: px.idxmin()].idxmax()
        tr = px.idxmin()
        row = {"episode": name}
        for k in ep_keys:
            sr_, e_ = strategies[k]
            w = sr_.loc[a:b].fillna(0)
            eq = (1 + w).cumprod()
            row[f"{k}|ret"] = eq.iloc[-1] - 1
            row[f"{k}|mdd"] = (eq / eq.cummax() - 1).min()
            row[f"{k}|pk2tr"] = (1 + sr_.loc[pk:tr].iloc[1:].fillna(0)).prod() - 1
        stw = st_base.loc[a:b]
        row["v2 avgExpo"] = to_expo(stw, CFG["expo"]).mean()
        row["%방어"] = (stw == 2).mean()
        row["first 방어"] = str(stw[stw == 2].index[0].date()) if (stw == 2).any() else "-"
        row["peak->trough"] = f"{pk.date()}..{tr.date()}"
        rows.append(row)
    ep = pd.DataFrame(rows).set_index("episode")
    for part in ["ret", "mdd", "pk2tr"]:
        sub = ep[[f"{k}|{part}" for k in ep_keys]].copy()
        sub.columns = ep_keys
        print(f"\n[{part}]")
        print(sub.map(lambda x: f"{x:+.1%}").to_string())
    print()
    print(ep[["v2 avgExpo", "%방어", "first 방어", "peak->trough"]].to_string(
        formatters={"v2 avgExpo": "{:.0%}".format, "%방어": "{:.0%}".format}))

    # ===================================================================== 5. BTC
    print(f"\n=== 5. BTC-USD (weekday grid; Monday return includes the weekend) {b0.date()} .. {b_end.date()} ===")
    bstr = {"Buy&hold": strat_returns(btc_r, one, 0)}
    sr, eu, st_b = run(asset="btc")
    bstr["v2 100/70/40"] = (sr, eu)
    bstr["v2 100/60/20"] = run(asset="btc", mapping=(1.0, 0.6, 0.2))[:2]
    bstr["MA200 100/40 (BTC's own)"] = strat_returns(btc_r, ma_btc, 1)
    bstr["VolTarget 15%"] = strat_returns(btc_r, vt_btc, 1)
    bstr["VolTarget 60%"] = strat_returns(btc_r, vt_btc60, 1)
    bstr["Composite-only 100/40"] = strat_returns(btc_r, ce, 1)
    avg_b = eu.loc[b0:b_end].mean()
    bstr[f"Constant {avg_b:.0%}"] = strat_returns(btc_r, pd.Series(avg_b, index=idx), 0)
    for b in ["L1_diag", "hedge"]:
        bstr[f"v2 - {b}"] = run({"drop": (b,)}, asset="btc")[:2]
    print(fmt_metrics({k: metrics(v[0], v[1], b0, b_end) for k, v in bstr.items()}).to_string())
    print("v2 state stats on BTC window:", state_stats(st_b, b0, b_end))
    bep = {"2015-16": ("2015-07-01", "2016-06-30"), "2018 bear": ("2018-01-02", "2018-12-31"),
           "2018Q4": ("2018-09-20", "2019-04-30"), "2020 Covid": ("2020-02-19", "2020-08-31"),
           "2021-05": ("2021-04-14", "2021-07-30"), "2022": ("2022-01-03", "2022-12-30"),
           "2023": ("2023-01-03", "2023-12-29"), "2025-04": ("2025-02-19", "2025-06-30"),
           "2025-10 liquidation": ("2025-10-01", "2025-12-31"), "2026-09/10": ("2026-08-19", str(b_end.date()))}
    rows = []
    for name, (a, b) in bep.items():
        row = {"episode": name}
        for k in ["Buy&hold", "v2 100/70/40", "MA200 100/40 (BTC's own)", "VolTarget 60%", "Composite-only 100/40"]:
            sr_, e_ = bstr[k]
            w = sr_.loc[a:b].fillna(0)
            eq = (1 + w).cumprod()
            row[k] = f"{eq.iloc[-1] - 1:+.0%} / {(eq / eq.cummax() - 1).min():+.0%}"
        row["v2 %방어"] = f"{(st_b.loc[a:b] == 2).mean():.0%}"
        rows.append(row)
    print("ret / mdd by episode:")
    print(pd.DataFrame(rows).set_index("episode").to_string())

    # ===================================================================== 6. forward risk by state
    print("\n=== 6. Does the state forecast risk? (state known at close t; forward window t+1..t+h, ^GSPC trading days) ===")
    sp = y["^GSPC"].loc[:SPX_END]
    traded = raw["^GSPC"].dropna().index
    spt = sp.reindex(traded).loc[s0:]
    rt = spt.pct_change(fill_method=None)
    fwd_vol = rt[::-1].rolling(21).std()[::-1].shift(-1) * np.sqrt(252)
    fwd_ret63 = spt.shift(-63) / spt - 1
    arr = spt.values
    fmdd = np.full(len(arr), np.nan)
    for i in range(len(arr) - 1):
        seg = arr[i:min(i + 64, len(arr))]
        fmdd[i] = seg.min() / seg[0] - 1
    fmdd = pd.Series(fmdd, index=spt.index)
    trail_vol = rt.rolling(20).std() * np.sqrt(252)
    F = pd.DataFrame({"fvol21": fwd_vol, "fmdd63": fmdd, "fret63": fwd_ret63, "trail_vol": trail_vol,
                      "v2": st_base.reindex(spt.index), "ma_below": (ma_spx.reindex(spt.index) < 1),
                      "comp": comp["valid"].reindex(spt.index)}).dropna(subset=["v2"])
    F["vol_tercile"] = pd.qcut(F["trail_vol"], 3, labels=["low", "mid", "high"])

    def summarise(g):
        return pd.Series({"share": len(g) / len(F), "fwd21 vol": g.fvol21.mean(), "fwd63 ret": g.fret63.mean(),
                          "fwd63 MDD med": g.fmdd63.median(), "P(MDD<-10%)": (g.fmdd63 < -0.10).mean()})

    tabs = []
    for col, name in [("v2", "v2 state"), ("ma_below", "below MA200"), ("comp", "composite valid"), ("vol_tercile", "trail vol tercile")]:
        tb = F.groupby(col, observed=True).apply(summarise)
        tb.index = [f"{name}={STATE_NAMES.get(i, i) if col == 'v2' else i}" for i in tb.index]
        tabs.append(tb)
    T = pd.concat(tabs)
    T.loc["ALL"] = summarise(F)
    print(T.to_string(formatters={"share": "{:.0%}".format, "fwd21 vol": "{:.1%}".format, "fwd63 ret": "{:+.1%}".format,
                                  "fwd63 MDD med": "{:+.1%}".format, "P(MDD<-10%)": "{:.0%}".format}))
    expo_map = {"v2": to_expo(st_base, CFG["expo"]), "MA200": ma_spx, "VolTarget": vt_spx}
    print("Spearman corr(exposure at t, forward 21d realised vol) -- more negative = better risk sizing:")
    for k, e in expo_map.items():
        ee = e.reindex(F.index)
        print(f"  {k:10s} {ee.corr(F['fvol21'], method='spearman'):+.2f}")
    print(f"  (trailing 20d vol vs fwd 21d vol: {F['trail_vol'].corr(F['fvol21'], method='spearman'):+.2f})")
    # incremental information beyond trailing vol: within each vol tercile, does v2 方어 still have higher forward risk?
    print("Within trailing-vol terciles: forward 21d vol / P(MDD<-10%) when flagged vs not:")
    for flag_name, flag in [("v2 방어", F.v2 == 2), ("below MA200", F.ma_below.astype(bool)), ("composite", F.comp.astype(bool))]:
        for tcl, g in F.groupby("vol_tercile", observed=True):
            fl = flag.loc[g.index]
            d, nd = g[fl], g[~fl]
            print(f"  {flag_name:11s} {tcl:4s}: flagged n={len(d):4d} vol {d.fvol21.mean():.1%} P {(d.fmdd63 < -0.1).mean():.0%} | "
                  f"other n={len(nd):4d} vol {nd.fvol21.mean():.1%} P {(nd.fmdd63 < -0.1).mean():.0%}")
    # mean-variance implied relative exposure by state: next-day returns while in each state
    nxt = spt.pct_change(fill_method=None).shift(-1)
    G = pd.DataFrame({"r": nxt, "v2": st_base.reindex(spt.index)}).dropna()
    mv = G.groupby("v2")["r"].agg(["mean", "var", "count"])
    mv["mu_ann"] = mv["mean"] * 252
    mv["vol_ann"] = np.sqrt(mv["var"] * 252)
    mv["mu/var"] = mv["mean"] / mv["var"]
    mv["implied expo vs 공격"] = mv["mu/var"] / mv["mu/var"].loc[0.0]
    mv["vol-parity expo"] = mv["vol_ann"].loc[0.0] / mv["vol_ann"]
    mv["mu_se"] = mv["vol_ann"] / np.sqrt(mv["count"] / 252)
    mv.index = [STATE_NAMES[i] for i in mv.index]
    print("Next-day S&P return by v2 state (mean-variance and vol-parity implied exposure, relative to 공격):")
    print(mv[["count", "mu_ann", "mu_se", "vol_ann", "implied expo vs 공격", "vol-parity expo"]].to_string(
        formatters={"mu_ann": "{:+.1%}".format, "mu_se": "±{:.1%}".format, "vol_ann": "{:.1%}".format,
                    "implied expo vs 공격": "{:.2f}".format, "vol-parity expo": "{:.2f}".format}))

    # ===================================================================== 7. placebo and bootstrap
    print("\n=== 7. Is the timing better than chance? ===")
    base_sr, base_e = strategies["v2 100/70/40"]
    r_w = spx_r.loc[s0:s_end]
    e_w = base_e.loc[s0:s_end].values
    rv = r_w.values
    ok = ~np.isnan(rv) & ~np.isnan(e_w)
    rv, e_w = rv[ok], e_w[ok]
    ppy = len(rv) / ((r_w.index[-1] - r_w.index[0]).days / 365.25)

    def stats_of(x):
        eq = np.cumprod(1 + x)
        return ann_sharpe(pd.Series(x), ppy), (eq / np.maximum.accumulate(eq) - 1).min()

    act = stats_of(e_w * rv)
    plc = []
    n = len(rv)
    for _ in range(1000):
        k = RNG.integers(252, n - 252)
        plc.append(stats_of(np.roll(e_w, k) * rv))
    plc = np.array(plc)
    print(f"Circular-shift placebo (1000 shifts of the v2 exposure path, same exposure mix and persistence):")
    print(f"  actual Sharpe {act[0]:.2f} vs placebo median {np.median(plc[:, 0]):.2f} "
          f"(actual beats {np.mean(plc[:, 0] < act[0]):.0%} of shifts); actual MaxDD {act[1]:+.1%} vs placebo median "
          f"{np.median(plc[:, 1]):+.1%} (actual shallower than {np.mean(plc[:, 1] < act[1]):.0%})")
    for k in ["MA200 100/40", "VolTarget 15%", "Composite-only 100/40"]:
        e2 = strategies[k][1].loc[s0:s_end].values[ok]
        a2 = stats_of(e2 * rv)
        p2 = np.array([stats_of(np.roll(e2, RNG.integers(252, n - 252)) * rv) for _ in range(500)])
        print(f"  {k:22s} Sharpe {a2[0]:.2f} beats {np.mean(p2[:, 0] < a2[0]):.0%} of its shifts; "
              f"MaxDD {a2[1]:+.1%} shallower than {np.mean(p2[:, 1] < a2[1]):.0%}")
    print("Stationary block bootstrap (mean block 63d, 1000 reps) of Sharpe difference vs buy-and-hold (rf=0):")
    bh = rv
    cands = {k: strategies[k][1].loc[s0:s_end].values[ok] for k in ["v2 100/70/40", "v2 100/60/20", "MA200 100/40", "VolTarget 15%", "Composite-only 100/40"]}
    diffs = {k: [] for k in cands}
    for _ in range(1000):
        ii = stationary_bootstrap_idx(n, 63, RNG)
        sb = ann_sharpe(pd.Series(bh[ii]), ppy)
        for k, e2 in cands.items():
            diffs[k].append(ann_sharpe(pd.Series(e2[ii] * rv[ii]), ppy) - sb)
    for k, d in diffs.items():
        d = np.array(d)
        pt = ann_sharpe(pd.Series(cands[k] * rv), ppy) - ann_sharpe(pd.Series(bh), ppy)
        print(f"  {k:22s} dSharpe {pt:+.2f}  90% CI [{np.percentile(d, 5):+.2f}, {np.percentile(d, 95):+.2f}]  P(>0) {np.mean(d > 0):.0%}")

    # ===================================================================== 8. credit proxy mapping notes
    print("\n=== 8. Credit / dollar proxy calibration facts ===")
    c20 = raw2["hyg_iei_20d"].loc[s0:s_end]
    print("Share of days HYG/IEI 20d log change <= threshold:",
          {f"{t:+.1%}": f"{(c20 <= t).mean():.1%}" for t in [-0.005, -0.01, -0.015, -0.02, -0.03, -0.04, -0.075]})
    lr = lambda k, a, b: np.log(y[k].loc[:b].iloc[-1] / y[k].loc[:a].iloc[-1])
    a, b = "2026-08-28", "2026-10-01"
    print(f"Anchor {a}->{b}: HY OAS ~260->~324bp (+25%, secondary sources); HYG/IEI {lr('HYG', a, b) - lr('IEI', a, b):+.2%}; "
          f"max 20d drop in Sep-Oct 2026 {c20.loc['2026-09-01':].min():+.2%}")
    d20 = raw2["dxy20"].loc[s0:s_end]
    print(f"Share of days DXY 20d >= +3%: {(d20 >= 0.03).mean():.1%}, >= +4%: {(d20 >= 0.04).mean():.1%}; "
          f"DXY 20d on 2026-10-13: {d20.loc[:'2026-10-13'].iloc[-1]:+.2%}")

    out = pd.concat([cols, l1[["label"]], raw2, comp, rawst.rename("raw_state"), st_base.rename("state")], axis=1).loc[s0:]
    out.to_csv(OUT / "states_daily.csv")
    print(f"\nDaily states written to {OUT / 'states_daily.csv'}")


if __name__ == "__main__":
    main()
