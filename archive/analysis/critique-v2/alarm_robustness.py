#!/usr/bin/env python3
"""Stress test of the v2 composite alarms (A: rate-led, B: flight-to-quality) and bucket rules.

Object under review: docs/v2-design.md (2026-10-14). The repo is read-only; this script only
reads analysis/.cache via load() from analysis/verify_v1.py and writes CSVs next to itself.

Definitions (exactly as v2 states them, on rolling 5-business-day simple % changes):
  A: HYG/IEI <= -0.5%  AND MOVE >= +10%  AND TLT <= -1%  AND DXY >= +0.5%
  B: HYG/IEI <= -0.5%  AND MOVE >= +10%  AND TLT >= +1%  AND VIX >= +15%
Business days = weekdays on which ^GSPC, HYG, ^MOVE or ^VIX printed (holidays dropped).
Episodes: start at the first firing; a new episode only after 20 business days without firing.
Outcome: max drawdown of ^GSPC closes over the next 63 business days, measured from the
firing close (min(close[t..t+63]) / close[t] - 1). BTC: same calendar window, 7-day closes.
Sample: firings from 2008-01-01. Windows that are not yet complete are excluded from rates.

Run:  python3 -I alarm_robustness.py   (about 15 seconds; writes episodes_v2.csv, grid_A.csv, grid_B.csv)
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import beta, spearmanr

REPO = Path("/home/user/macro-risk-dashboard")
OUT = Path(__file__).resolve().parent
H = 63          # forward window (business days)
GAP = 20        # de-clustering gap
VALID = 10      # v2 validity window for composite alarms
START = "2008-01-01"
SPLIT = "2017-01-01"
DD = -0.10      # drawdown event threshold
BTC_DD = -0.20
NBOOT = 10000
RNG = np.random.default_rng(20261014)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 200)


def section(t):
    print(f"\n{'=' * 100}\n{t}\n{'=' * 100}")


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------
spec = importlib.util.spec_from_file_location("verify_v1", REPO / "analysis" / "verify_v1.py")
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)
nom, tips, acm, y = V.load()

raw_all = pd.read_csv(V.CACHE / "yahoo.csv", index_col=0, parse_dates=True)
raw = raw_all[raw_all.index.dayofweek < 5]
bd = raw.index[raw[["^GSPC", "HYG", "^MOVE", "^VIX"]].notna().any(axis=1)]
P = y.loc[bd].copy()
N = len(bd)
s0 = int(bd.searchsorted(pd.Timestamp(START)))
split_pos = int(bd.searchsorted(pd.Timestamp(SPLIT)))

spx = P["^GSPC"].to_numpy(float)
win = sliding_window_view(spx, H + 1)
MDD = np.full(N, np.nan)
MDD[: N - H] = win.min(axis=1) / spx[: N - H] - 1
MDD_PART = np.array([spx[i:].min() / spx[i] - 1 for i in range(N)])  # partial window (for live episodes)
FRET = np.full(N, np.nan)
FRET[: N - H] = spx[H:] / spx[: N - H] - 1
lr = np.diff(np.log(spx))
FVOL = np.full(N, np.nan)
FVOL[: N - H] = sliding_window_view(lr, H).std(axis=1, ddof=1) * np.sqrt(252)
DD252 = (P["^GSPC"] / P["^GSPC"].rolling(252, min_periods=1).max() - 1).to_numpy()

btc = raw_all["BTC-USD"].dropna()
btc_idx = btc.index.values
btc_v = btc.to_numpy(float)
BTC_MDD = np.full(N, np.nan)
for i in range(N - H):
    a = np.searchsorted(btc_idx, bd[i].to_datetime64())
    if a >= len(btc_idx) or btc_idx[a] != bd[i].to_datetime64():
        continue
    b = np.searchsorted(btc_idx, bd[i + H].to_datetime64(), side="right")
    BTC_MDD[i] = btc_v[a:b].min() / btc_v[a] - 1

# features (rolling simple % changes on business-day rows)
F = pd.DataFrame(index=bd)
F["cr5"] = (P["HYG"] / P["IEI"]).pct_change(5)
F["cr20"] = (P["HYG"] / P["IEI"]).pct_change(20)
F["mv5"] = P["^MOVE"].pct_change(5)
F["tlt5"] = P["TLT"].pct_change(5)
F["dxy5"] = P["DX-Y.NYB"].pct_change(5)
F["dxy20"] = P["DX-Y.NYB"].pct_change(20)
F["vix5"] = P["^VIX"].pct_change(5)
F["tyx5"] = P["^TYX"].diff(5)        # ^TYX is quoted in percent -> 0.25 = 25bp
F["tyx20"] = P["^TYX"].diff(20)
F["vvr"] = P["^VIX"] / P["^VIX3M"]
F["move"] = P["^MOVE"]
F["below200"] = P["^GSPC"] < P["^GSPC"].rolling(200).mean()
r = np.log(P[["^GSPC", "TLT"]]).diff()
F["sbcorr"] = r["^GSPC"].rolling(63).corr(r["TLT"])
A_ = {k: F[k].to_numpy(float) for k in F.columns if k != "below200"}

INSAMPLE = np.zeros(N, bool)
INSAMPLE[s0:] = True
ELIG = np.flatnonzero(INSAMPLE & ~np.isnan(MDD))  # start dates with complete windows, 2008+


def nz(x):
    return np.nan_to_num(x, nan=0.0)


def sig_A(cr=-0.005, mv=0.10, tlt=-0.01, dxy=0.005):
    with np.errstate(invalid="ignore"):
        return (A_["cr5"] <= cr) & (A_["mv5"] >= mv) & (A_["tlt5"] <= tlt) & (A_["dxy5"] >= dxy) & INSAMPLE


def sig_B(cr=-0.005, mv=0.10, tlt=0.01, vix=0.15):
    with np.errstate(invalid="ignore"):
        return (A_["cr5"] <= cr) & (A_["mv5"] >= mv) & (A_["tlt5"] >= tlt) & (A_["vix5"] >= vix) & INSAMPLE


def episodes(sig, gap=GAP):
    pos = np.flatnonzero(sig)
    if len(pos) == 0:
        return np.array([], int)
    keep = np.r_[True, np.diff(pos) > gap]
    return pos[keep]


def active(sig, k=VALID):
    """True on days within k business days after (and including) a firing day."""
    s = pd.Series(sig.astype(float))
    return (s.rolling(k, min_periods=1).max() > 0).to_numpy()


def rate(starts, dd=DD, arr=None):
    arr = MDD if arr is None else arr
    m = arr[starts]
    m = m[~np.isnan(m)]
    return (m < dd).mean() if len(m) else np.nan, len(m)


def summarize(starts, label=""):
    m = MDD[starts]
    ok = ~np.isnan(m)
    mb = BTC_MDD[starts]
    okb = ~np.isnan(mb)
    return {
        "signal": label,
        "episodes": len(starts),
        "complete": int(ok.sum()),
        "P(MDD<-10%)": (m[ok] < DD).mean() if ok.any() else np.nan,
        "P(MDD<-5%)": (m[ok] < -0.05).mean() if ok.any() else np.nan,
        "med MDD": np.median(m[ok]) if ok.any() else np.nan,
        "mean MDD": m[ok].mean() if ok.any() else np.nan,
        "mean fwd63": np.nanmean(FRET[starts]) if ok.any() else np.nan,
        "med fwdvol": np.nanmedian(FVOL[starts]) if ok.any() else np.nan,
        "med DD@fire": np.median(DD252[starts]) if len(starts) else np.nan,
        "BTC n": int(okb.sum()),
        "BTC med MDD": np.median(mb[okb]) if okb.any() else np.nan,
        "P(BTC MDD<-20%)": (mb[okb] < BTC_DD).mean() if okb.any() else np.nan,
    }


def fmt(df):
    df = df.copy()
    for c in df.columns:
        if c.startswith("P(") or "MDD" in c or "fwd" in c or "DD@" in c:
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:+.1%}" if ("MDD" in c and not c.startswith("P(")) or "fwd63" in c or "DD@" in c else f"{v:.0%}" if c.startswith("P(") else f"{v:.1%}")
    return df


def uncond(lo=s0, hi=N, arr=None, dd=DD):
    """Unconditional rate: all start days and non-overlapping (every 63rd day, averaged over offsets)."""
    arr = MDD if arr is None else arr
    e = np.arange(lo, hi)
    e = e[~np.isnan(arr[e])]
    p_all = (arr[e] < dd).mean()
    p_off = [(arr[e[o::H]] < dd).mean() for o in range(H)]
    return p_all, float(np.mean(p_off)), len(e[0::H]), e


def boot_uncond(e, arr=None, dd=DD, reps=NBOOT):
    arr = MDD if arr is None else arr
    out = np.empty(reps)
    for b in range(reps):
        o = RNG.integers(H)
        w = arr[e[o::H]] < dd
        out[b] = w[RNG.integers(0, len(w), len(w))].mean()
    return out


def boot_eps(starts, arr=None, dd=DD, reps=NBOOT):
    arr = MDD if arr is None else arr
    m = arr[starts]
    m = m[~np.isnan(m)] < dd
    idx = RNG.integers(0, len(m), (reps, len(m)))
    return m[idx].mean(axis=1)


def clopper(k, n, a=0.05):
    lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def spaced_random_p(n_ep, e, arr=None, dd=DD, reps=5000, gap=GAP):
    """Null: P(MDD<dd) for n_ep random start dates drawn from e with the same >gap spacing."""
    arr = MDD if arr is None else arr
    out = np.empty(reps)
    for b in range(reps):
        perm = RNG.permutation(e)
        chosen = []
        for p in perm:
            if all(abs(p - c) > gap for c in chosen):
                chosen.append(p)
                if len(chosen) == n_ep:
                    break
        out[b] = (arr[np.array(chosen)] < dd).mean()
    return out


def ols_nw(yv, X, lags=H):
    X = np.column_stack([np.ones(len(yv)), X])
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ yv
    u = yv - X @ b
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for l in range(1, lags + 1):
        G = Xu[l:].T @ Xu[:-l]
        S += (1 - l / (lags + 1)) * (G + G.T)
    Vc = XtXi @ S @ XtXi
    return b, np.sqrt(np.diag(Vc))


# ----------------------------------------------------------------------------------------------
# 1. rolling-5-day recomputation with de-clustering
# ----------------------------------------------------------------------------------------------
section("1. v2 composite alarms on ROLLING 5-business-day changes, de-clustered (gap 20bd), 63bd forward")
print(f"business days {bd[0].date()}..{bd[-1].date()}; sample firings from {START}; last complete 63bd window starts "
      f"{bd[N - H - 1].date()}")
p_all, p_no, n_no, E_ALL = uncond()
pb_all, pb_no, _, EB = uncond(lo=int(bd.searchsorted(pd.Timestamp('2014-09-17'))), arr=BTC_MDD, dd=BTC_DD)
print(f"Unconditional S&P P(MDD63<-10%): all start days {p_all:.1%} (n={len(E_ALL)}), "
      f"non-overlapping 63bd starts {p_no:.1%} (~{n_no} windows per offset); median MDD {np.median(MDD[E_ALL]):+.1%}; "
      f"P(MDD<-5%) {(MDD[E_ALL] < -0.05).mean():.0%}")
print(f"Unconditional BTC (2014-09+) P(MDD63<-20%): all days {pb_all:.1%}, non-overlapping {pb_no:.1%}; "
      f"median BTC MDD {np.median(BTC_MDD[EB]):+.1%}")

SA, SB = sig_A(), sig_B()
EA, EBp = episodes(SA), episodes(SB)
rows = [summarize(E_ALL, "unconditional (all days)"),
        summarize(EA, "A rate-led (v2)"), summarize(EBp, "B flight-to-quality (v2)"),
        summarize(episodes(SA | SB), "A or B")]
# also non-declustered firing days for reference
fa, fb = np.flatnonzero(SA), np.flatnonzero(SB)
rows += [summarize(fa, "A all firing days (not declustered)"), summarize(fb, "B all firing days (not declustered)")]
T1 = pd.DataFrame(rows)
print(fmt(T1).to_string(index=False))
print(f"\nA firing days: {SA.sum()} ({SA[s0:].mean():.2%} of days); B firing days: {SB.sum()} ({SB[s0:].mean():.2%})")


def ep_table(starts, sig, name):
    rows = []
    for s in starts:
        # days firing within the episode
        e = s
        pos = np.flatnonzero(sig[s:]) + s
        last = s
        for p in pos:
            if p - last > GAP:
                break
            last = p
        nfire = int(sig[s:last + 1].sum())
        rows.append({"alarm": name, "start": bd[s].date(), "last fire": bd[last].date(), "fire days": nfire,
                     "SPX DD@fire": DD252[s], "SPX MDD63": MDD[s] if not np.isnan(MDD[s]) else MDD_PART[s],
                     "complete": not np.isnan(MDD[s]), "SPX fwd63": FRET[s], "BTC MDD63": BTC_MDD[s]})
    return pd.DataFrame(rows)


EPS = pd.concat([ep_table(EA, SA, "A"), ep_table(EBp, SB, "B")])
EPS.to_csv(OUT / "episodes_v2.csv", index=False)
show = EPS.copy()
for c in ["SPX DD@fire", "SPX MDD63", "SPX fwd63", "BTC MDD63"]:
    show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:+.1%}")
print("\nEpisode list (MDD for incomplete windows is the partial MDD to date):")
print(show.to_string(index=False))

# ----------------------------------------------------------------------------------------------
# 4 (moved up). uncertainty
# ----------------------------------------------------------------------------------------------
section("4. Uncertainty: episode bootstrap CIs for P(MDD63<-10%) vs unconditional (non-overlapping windows)")
ub = boot_uncond(E_ALL)
print(f"unconditional non-overlapping: point {p_no:.1%}, bootstrap 95% CI [{np.percentile(ub, 2.5):.1%}, "
      f"{np.percentile(ub, 97.5):.1%}]")
out = []
for name, st in [("A", EA), ("B", EBp), ("A or B", episodes(SA | SB))]:
    p, n = rate(st)
    k = int(round(p * n))
    bb = boot_eps(st)
    diff = bb - ub
    null = spaced_random_p(n, E_ALL)
    lo, hi = clopper(k, n)
    out.append([name, n, k, f"{p:.0%}", f"[{np.percentile(bb, 2.5):.0%}, {np.percentile(bb, 97.5):.0%}]",
                f"[{lo:.0%}, {hi:.0%}]", f"{p - p_no:+.0%}",
                f"[{np.percentile(diff, 2.5):+.0%}, {np.percentile(diff, 97.5):+.0%}]",
                f"{(diff <= 0).mean():.3f}", f"{(null >= p).mean():.3f}"])
print(pd.DataFrame(out, columns=["alarm", "n complete", "k hits", "P", "boot 95% CI", "Clopper-Pearson 95%",
                                 "lift vs uncond", "boot CI of lift", "boot P(lift<=0)",
                                 "perm p (spaced random dates)"]).to_string(index=False))
# BTC
print("\nBTC (2014-09+): P(BTC MDD63 < -20%)")
ubb = boot_uncond(EB, arr=BTC_MDD, dd=BTC_DD)
out = []
for name, st in [("A", EA), ("B", EBp)]:
    st2 = st[~np.isnan(BTC_MDD[st])]
    p, n = rate(st2, dd=BTC_DD, arr=BTC_MDD)
    bb = boot_eps(st2, arr=BTC_MDD, dd=BTC_DD)
    null = spaced_random_p(n, EB, arr=BTC_MDD, dd=BTC_DD, reps=3000)
    out.append([name, n, f"{p:.0%}", f"[{np.percentile(bb, 2.5):.0%}, {np.percentile(bb, 97.5):.0%}]",
                f"{pb_no:.0%} [{np.percentile(ubb, 2.5):.0%}, {np.percentile(ubb, 97.5):.0%}]",
                f"{(null >= p).mean():.3f}", f"{np.median(BTC_MDD[st2]):+.1%}", f"{np.median(BTC_MDD[EB]):+.1%}"])
print(pd.DataFrame(out, columns=["alarm", "n", "P", "boot CI", "uncond (CI)", "perm p", "med BTC MDD",
                                 "uncond med"]).to_string(index=False))

# ----------------------------------------------------------------------------------------------
# 3. sensitivity
# ----------------------------------------------------------------------------------------------
section("3. Sensitivity: one threshold at a time scaled by magnitude x0.5/x0.75/x1/x1.25/x1.5 (x>1 = stricter)")
base_A = dict(cr=-0.005, mv=0.10, tlt=-0.01, dxy=0.005)
base_B = dict(cr=-0.005, mv=0.10, tlt=0.01, vix=0.15)
mults = [0.5, 0.75, 1.0, 1.25, 1.5]
for name, base, fn in [("A", base_A, sig_A), ("B", base_B, sig_B)]:
    rows = []
    for k in list(base) + ["ALL"]:
        row_n, row_p, row_m = [], [], []
        for m in mults:
            kw = dict(base)
            if k == "ALL":
                kw = {kk: vv * m for kk, vv in base.items()}
            else:
                kw[k] = base[k] * m
            st = episodes(fn(**kw))
            p, n = rate(st)
            row_n.append(len(st))
            row_p.append(p)
            row_m.append(np.nanmedian(MDD[st]) if len(st) else np.nan)
        rows.append([k] + [f"{a:>3d} / {'  -' if pd.isna(b) else f'{b:3.0%}'} / {'' if pd.isna(c) else f'{c:+.1%}'}"
                           for a, b, c in zip(row_n, row_p, row_m)])
    print(f"\n{name}: cells = episodes / P(MDD<-10%) / median MDD    (unconditional {p_no:.0%}, median {np.median(MDD[E_ALL]):+.1%})")
    print(pd.DataFrame(rows, columns=["threshold"] + [f"x{m}" for m in mults]).to_string(index=False))

# ----------------------------------------------------------------------------------------------
# 2. out-of-sample
# ----------------------------------------------------------------------------------------------
section("2. Out-of-sample: pick thresholds on one half (small grid), evaluate on the other")
TRAIN_MASKS = {"2008-2016": (s0, split_pos), "2017-2026": (split_pos, N)}
gridA = dict(cr=[-0.0025, -0.005, -0.0075, -0.01, -0.015], mv=[0.05, 0.10, 0.15, 0.20, 0.25],
             tlt=[-0.005, -0.01, -0.015, -0.02], dxy=[0.0, 0.0025, 0.005, 0.0075, 0.01])
gridB = dict(cr=[-0.0025, -0.005, -0.0075, -0.01, -0.015], mv=[0.05, 0.10, 0.15, 0.20, 0.25],
             tlt=[0.005, 0.01, 0.015, 0.02], vix=[0.05, 0.10, 0.15, 0.20, 0.30])
MIN_EP = 5
SHRINK = 4


def period_stats(st, lo, hi):
    st = st[(st >= lo) & (st < hi)]
    m = MDD[st]
    m = m[~np.isnan(m)]
    return len(m), ((m < DD).mean() if len(m) else np.nan), (m.mean() if len(m) else np.nan)


for name, grid, fn, base in [("A", gridA, sig_A, base_A), ("B", gridB, sig_B, base_B)]:
    keys = list(grid)
    combos = pd.MultiIndex.from_product([grid[k] for k in keys], names=keys)
    res = []
    for c in combos:
        st = episodes(fn(**dict(zip(keys, c))))
        rec = dict(zip(keys, c))
        for per, (lo, hi) in TRAIN_MASKS.items():
            n, p, mm = period_stats(st, lo, hi)
            rec[f"n {per}"], rec[f"P {per}"], rec[f"meanMDD {per}"] = n, p, mm
        res.append(rec)
    R = pd.DataFrame(res)
    R.to_csv(OUT / f"grid_{name}.csv", index=False)
    print(f"\n--- alarm {name}: {len(R)} grid cells ---")
    for tr, te in [("2008-2016", "2017-2026"), ("2017-2026", "2008-2016")]:
        lo, hi = TRAIN_MASKS[tr]
        p0_tr = (MDD[np.arange(lo, hi)][~np.isnan(MDD[np.arange(lo, hi)])] < DD).mean()
        lo2, hi2 = TRAIN_MASKS[te]
        mte = MDD[np.arange(lo2, hi2)]
        mte = mte[~np.isnan(mte)]
        p0_te, mm0_te = (mte < DD).mean(), mte.mean()
        ok = R[f"n {tr}"] >= MIN_EP
        score = (R[f"P {tr}"] * R[f"n {tr}"] + SHRINK * p0_tr) / (R[f"n {tr}"] + SHRINK)
        best = R[ok].assign(score=score[ok]).sort_values(["score", f"n {tr}"], ascending=False)
        top = best.iloc[0]
        # v2 thresholds
        v2row = R.loc[np.all([np.isclose(R[k], base[k]) for k in keys], axis=0)].iloc[0]
        # median performance of the top-10 train cells in test
        top10 = best.head(10)
        both = R[(R[f"n {tr}"] >= 3) & (R[f"n {te}"] >= 3)]
        rho = spearmanr(both[f"P {tr}"], both[f"P {te}"]).statistic
        rho_m = spearmanr(both[f"meanMDD {tr}"], both[f"meanMDD {te}"]).statistic
        # mean-MDD objective
        sc2 = (R[f"meanMDD {tr}"] * R[f"n {tr}"] + SHRINK * MDD[np.arange(lo, hi)][~np.isnan(MDD[np.arange(lo, hi)])].mean()) / (R[f"n {tr}"] + SHRINK)
        best2 = R[ok].assign(score=sc2[ok]).sort_values("score").iloc[0]
        test_p = R[f"P {te}"][R[f"n {te}"] >= 3]
        print(f"train {tr} (uncond P {p0_tr:.0%}) -> test {te} (uncond P {p0_te:.0%}, mean MDD {mm0_te:+.1%})")
        print(f"  best by shrunk P:   {dict((k, top[k]) for k in keys)}  train n={int(top[f'n {tr}'])} P={top[f'P {tr}']:.0%}"
              f"  | test n={int(top[f'n {te}'])} P={top[f'P {te}']:.0%} meanMDD={top[f'meanMDD {te}']:+.1%}"
              f"  (test-P percentile among grid cells with n>=3: {(test_p < top[f'P {te}']).mean():.0%})")
        print(f"  best by mean MDD:   {dict((k, best2[k]) for k in keys)}  train n={int(best2[f'n {tr}'])} meanMDD={best2[f'meanMDD {tr}']:+.1%}"
              f"  | test n={int(best2[f'n {te}'])} P={best2[f'P {te}']:.0%} meanMDD={best2[f'meanMDD {te}']:+.1%}")
        print(f"  top-10 train cells in test: median n={top10[f'n {te}'].median():.0f}, median P={top10[f'P {te}'].median():.0%},"
              f" median meanMDD={top10[f'meanMDD {te}'].median():+.1%}")
        print(f"  v2 thresholds:      train n={int(v2row[f'n {tr}'])} P={v2row[f'P {tr}']:.0%} | test n={int(v2row[f'n {te}'])} "
              f"P={v2row[f'P {te}']:.0%} meanMDD={v2row[f'meanMDD {te}']:+.1%}")
        print(f"  Spearman(train, test) across {len(both)} cells with >=3 eps in both halves: P {rho:+.2f}, mean MDD {rho_m:+.2f}")
        print(f"  share of cells (n>=3 in test) with test P > test uncond: {(test_p > p0_te).mean():.0%}")

# ----------------------------------------------------------------------------------------------
# 5. single-indicator alarms and incremental value
# ----------------------------------------------------------------------------------------------
section("5. Single-indicator alarms vs composites (same de-clustering and outcome)")
with np.errstate(invalid="ignore"):
    SINGLES = {
        "MOVE 5d >= +20%": (A_["mv5"] >= 0.20) & INSAMPLE,
        "HYG/IEI 5d <= -1%": (A_["cr5"] <= -0.01) & INSAMPLE,
        "VIX/VIX3M > 1": (A_["vvr"] > 1.0) & INSAMPLE,
        "S&P < 200d SMA": F["below200"].to_numpy() & INSAMPLE,
        "^TYX 20d >= +40bp": (A_["tyx20"] >= 0.40) & INSAMPLE,
    }
ALL_SIG = {"A (v2)": SA, "B (v2)": SB, **SINGLES}
rows = [summarize(E_ALL, "unconditional")]
for k, s in ALL_SIG.items():
    d = summarize(episodes(s), k)
    d["share days"] = s[s0:].mean()
    rows.append(d)
T5 = pd.DataFrame(rows)
T5["share days"] = T5["share days"].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
print(fmt(T5).to_string(index=False))

print("\nCo-firing: share of A/B episode starts at which each single alarm was active (fired within the prior 10bd incl. same day)")
ACT = {k: active(s) for k, s in ALL_SIG.items()}
ACT["S&P < 200d SMA"] = SINGLES["S&P < 200d SMA"]  # a state already
rows = []
for nm, st in [("A", EA), ("B", EBp)]:
    rows.append([nm] + [f"{ACT[k][st].mean():.0%}" for k in SINGLES] +
                [f"{np.mean([any(ACT[k][s] for k in SINGLES) for s in st]):.0%}"])
print(pd.DataFrame(rows, columns=["alarm"] + list(SINGLES) + ["any single"]).to_string(index=False))

print("\nDaily-state regressions (2008+, complete windows): y = 1{S&P MDD63 < -10%} or MDD63; regressors = alarm active "
      "(10bd validity; S&P<200d as state). OLS, Newey-West(63) s.e.")
idx = ELIG
yb = (MDD[idx] < DD).astype(float)
ym = MDD[idx]
names = list(ACT)
X = np.column_stack([ACT[k][idx].astype(float) for k in names])
rows = []
for j, k in enumerate(names):
    b1, se1 = ols_nw(yb, X[:, [j]])
    b2, se2 = ols_nw(ym, X[:, [j]])
    rows.append([k, f"{X[:, j].mean():.1%}", f"{b1[1]:+.3f} ({b1[1] / se1[1]:+.1f})", f"{b2[1] * 100:+.2f} ({b2[1] / se2[1]:+.1f})"])
bm1, sm1 = ols_nw(yb, X)
bm2, sm2 = ols_nw(ym, X)
for j, k in enumerate(names):
    rows[j] += [f"{bm1[j + 1]:+.3f} ({bm1[j + 1] / sm1[j + 1]:+.1f})", f"{bm2[j + 1] * 100:+.2f} ({bm2[j + 1] / sm2[j + 1]:+.1f})"]
print(pd.DataFrame(rows, columns=["regressor", "share days active", "univ dP (t)", "univ dMDD pp (t)",
                                  "multi dP (t)", "multi dMDD pp (t)"]).to_string(index=False))
print(f"(intercepts: univariate base P {yb.mean():.1%}; multivariate intercept P {bm1[0]:.1%}, MDD {bm2[0]:+.1%})")

print("\nConditional on each single alarm being active: does A-or-B active add anything? (daily, complete windows)")
AB = ACT["A (v2)"] | ACT["B (v2)"]
rows = []
for k in SINGLES:
    m = ACT[k][idx]
    for flag, lab in [(True, "with A|B"), (False, "without A|B")]:
        mm = m & (AB[idx] == flag)
        rows.append([k, lab, int(mm.sum()), f"{(MDD[idx][mm] < DD).mean():.0%}" if mm.any() else "",
                     f"{np.median(MDD[idx][mm]):+.1%}" if mm.any() else ""])
print(pd.DataFrame(rows, columns=["single alarm active", "", "days", "P(MDD<-10%)", "med MDD"]).to_string(index=False))

# composite episodes that occurred when no single alarm (except below200) was active
print("\nA/B episodes NOT accompanied by MOVE20, HYG/IEI-1%, VIX/VIX3M>1 or TYX40 within 10bd:")
fast = ["MOVE 5d >= +20%", "HYG/IEI 5d <= -1%", "VIX/VIX3M > 1", "^TYX 20d >= +40bp"]
for nm, st in [("A", EA), ("B", EBp)]:
    solo = [s for s in st if not any(ACT[k][s] for k in fast)]
    print(f"  {nm}: {len(solo)} of {len(st)} -> " + ", ".join(
        f"{bd[s].date()} (MDD {MDD[s]:+.1%})" if not np.isnan(MDD[s]) else f"{bd[s].date()} (partial {MDD_PART[s]:+.1%})" for s in solo))

# ----------------------------------------------------------------------------------------------
# 6. bucket rules (computable subset)
# ----------------------------------------------------------------------------------------------
section("6. v2 bucket yellow/red rules computable from cached data (daily, 2008+)")
# Layer-1 diagnosis from GSW + ACM, 20bd changes in bp
D = pd.DataFrame({"n": nom["SVENY10"], "real": tips["TIPSY10"], "be": tips["BKEVEN10"],
                  "tp": acm["ACMTP10"], "ep": acm["ACMRNY10"]}).dropna()
d20 = D.diff(20) * 100
spx4 = P["^GSPC"].reindex(D.index, method="ffill").pct_change(20)
real_lvl = D["real"]
real_pct = real_lvl.rolling(756).apply(lambda a: (a <= a[-1]).mean() * 100, raw=True)


def diag_row(dn, dr, db, dt, de, s):
    up_s, dn_s = s >= 0.02, s <= -0.02
    if abs(dn) < 20 and abs(db) < 15:
        return "no verdict", "G"
    if abs(dn) < 20:
        if db >= 15 and dr <= -20:
            return "easing+reflation", "G"
        if db <= -15 and dr >= 20:
            return "disinflation+real tightening", "Y"
        return "UNMAPPED flat-nominal (BE moved, real ->)", "Y"
    d1 = "EP" if abs(de) > abs(dt) else "TP"
    d2 = "real" if abs(dr) > abs(db) else "BE"
    if dn >= 20:
        if d1 == "EP" and d2 == "real":
            return ("growth repricing", "G") if up_s else ("policy shock", "Y") if dn_s else ("tightening, boundary", "Y")
        return {("EP", "BE"): ("inflation-response tightening", "Y"), ("TP", "real"): ("fiscal/supply premium", "Y"),
                ("TP", "BE"): ("inflation-uncertainty premium", "Y")}[(d1, d2)]
    if d1 == "EP" and d2 == "real":
        return ("easing rally", "G") if up_s else ("growth scare", "Y") if dn_s else ("UNMAPPED falling EP/real, S&P ->", "Y")
    return {("EP", "BE"): ("disinflation easing", "G"), ("TP", "real"): ("safe-haven demand", "Y"),
            ("TP", "BE"): ("inflation-risk reduction", "G")}[(d1, d2)]


dd_ = pd.concat([d20, spx4.rename("s")], axis=1).dropna()
diag = pd.DataFrame([diag_row(r.n, r.real, r.be, r.tp, r.ep, r.s) for r in dd_.itertuples()],
                    index=dd_.index, columns=["verdict", "color"])
diag["realpct"] = real_pct.reindex(diag.index)
diag.loc[(diag.color == "Y") & (diag.realpct >= 95), "color"] = "R"
diag_bd = diag.reindex(bd, method="ffill", limit=3)

mv5, tyx5, tyx20, vvr, sbc, dxy20, mlev = (A_[k] for k in ["mv5", "tyx5", "tyx20", "vvr", "sbcorr", "dxy20", "move"])
with np.errstate(invalid="ignore"):
    rules = {
        "RateVol Y: MOVE 5d >= +20%": mv5 >= 0.20,
        "RateVol Y: TYX 20d >= +40bp": tyx20 >= 0.40,
        "RateVol R: TYX 5d>=+25bp & MOVE 5d>=+20%": (tyx5 >= 0.25) & (mv5 >= 0.20),
        "RateVol R: MOVE > 126": mlev > 126,
        "EqVol Y: VIX/VIX3M > 0.95 (incl. >1)": vvr > 0.95,
        "EqVol R: VIX/VIX3M > 1.0": vvr > 1.0,
        "Hedge Y: SPX-TLT corr63 > 0 (incl. >0.3)": sbc > 0,
        "Hedge R: SPX-TLT corr63 > 0.3": sbc > 0.3,
        "Fund Y (proxy): DXY 20d >= +3%": dxy20 >= 0.03,
        "Diag Y-or-R": diag_bd["color"].isin(["Y", "R"]).to_numpy(),
        "Diag R (Y & real >= 3y p95)": (diag_bd["color"] == "R").to_numpy(),
        "Composite A|B valid (10bd)": AB,
    }
bucket = pd.DataFrame(index=bd)
bucket["ratevol"] = np.where(rules["RateVol R: TYX 5d>=+25bp & MOVE 5d>=+20%"] | rules["RateVol R: MOVE > 126"], 2,
                             np.where(rules["RateVol Y: MOVE 5d >= +20%"] | rules["RateVol Y: TYX 20d >= +40bp"], 1, 0))
bucket["eqvol"] = np.where(vvr > 1.0, 2, np.where(vvr > 0.95, 1, 0))
bucket["hedge"] = np.where(sbc > 0.3, 2, np.where(sbc > 0, 1, 0))
bucket["fund"] = np.where(dxy20 >= 0.03, 1, 0)
bucket["diag"] = diag_bd["color"].map({"G": 0, "Y": 1, "R": 2}).fillna(0).to_numpy()
nred = (bucket == 2).sum(axis=1).to_numpy()
nyel = (bucket == 1).sum(axis=1).to_numpy()
state = np.where((nred >= 2) | AB, "defensive", np.where((nred >= 1) | (nyel >= 2), "neutral", "aggressive"))
rules["STATE aggressive"] = state == "aggressive"
rules["STATE neutral"] = state == "neutral"
rules["STATE defensive"] = state == "defensive"
rules["STATE defensive w/o composite (>=2 red)"] = nred >= 2

base_m = MDD[idx]
rows = [["unconditional", "100%", len(idx), "", f"{(base_m < DD).mean():.0%}", f"{(base_m < -0.05).mean():.0%}",
         f"{np.median(base_m):+.1%}", f"{np.percentile(base_m, 10):+.1%}", f"{np.nanmean(FRET[idx]):+.1%}",
         f"{np.nanmedian(FVOL[idx]):.1%}", "", ""]]
for k, s in rules.items():
    s = np.asarray(s, bool)
    m = s[idx]
    if m.sum() == 0:
        continue
    b1, se1 = ols_nw(yb, m.astype(float)[:, None])
    bv, sev = ols_nw(FVOL[idx], m.astype(float)[:, None])
    rows.append([k, f"{s[s0:].mean():.1%}", int(m.sum()), len(episodes(s & INSAMPLE)), f"{(base_m[m] < DD).mean():.0%}",
                 f"{(base_m[m] < -0.05).mean():.0%}", f"{np.median(base_m[m]):+.1%}", f"{np.percentile(base_m[m], 10):+.1%}",
                 f"{np.nanmean(FRET[idx][m]):+.1%}", f"{np.nanmedian(FVOL[idx][m]):.1%}",
                 f"{b1[1] / se1[1]:+.1f}", f"{bv[1] / sev[1]:+.1f}"])
T6 = pd.DataFrame(rows, columns=["rule", "share days", "days (complete)", "episodes", "P(MDD<-10%)", "P(MDD<-5%)",
                                 "med MDD", "p10 MDD", "mean fwd63", "med fwd vol", "NW t dP", "NW t dVol"])
print(T6.to_string(index=False))
print("\nNote: credit bucket (HY OAS, CCC-BB) and NFCI / SOFR-IORB / broad dollar are not computable here (FRED blocked);"
      " fund bucket uses DXY 20d as a proxy, credit bucket is treated as green, so the state is biased toward 'aggressive'.")

print("\nDiagnosis verdict frequencies (daily, 2008+):")
dv = diag.loc[START:, "verdict"].value_counts(normalize=True)
print(dv.map(lambda v: f"{v:.1%}").to_string())

print("\nShare of days by year: hedge bucket red, defensive state, composite valid, diag red")
yr = pd.DataFrame({"hedge red": sbc > 0.3, "hedge>=yellow": sbc > 0, "diag red": rules["Diag R (Y & real >= 3y p95)"],
                   "defensive": state == "defensive", "neutral": state == "neutral", "A|B valid": AB}, index=bd).loc[START:]
print(yr.groupby(yr.index.year).mean().map(lambda v: f"{v:.0%}").to_string())

print("\nCurrent readings (last rows):")
cur = pd.DataFrame({"MOVE": mlev, "MOVE5d": mv5, "TYX5d(bp)": tyx5 * 100, "TYX20d(bp)": tyx20 * 100, "VIX/VIX3M": vvr,
                    "SPX-TLT corr63": sbc, "DXY20d": dxy20, "HYG/IEI 5d": A_["cr5"], "TLT5d": A_["tlt5"], "DXY5d": A_["dxy5"],
                    "A": SA, "B": SB, "A|B valid": AB, "diag": diag_bd["verdict"].to_numpy(),
                    "diag color": diag_bd["color"].to_numpy(), "state": state}, index=bd)
print(cur.loc["2026-09-15":].round(3).to_string())

# ----------------------------------------------------------------------------------------------
# extra checks
# ----------------------------------------------------------------------------------------------
section("X1. Bridge from v2's quoted weekly numbers to the daily rolling-5d definition")
w = y[["HYG", "IEI", "^MOVE", "TLT", "DX-Y.NYB", "^GSPC", "^VIX"]].ffill().resample("W-FRI").last().loc["2008":]
L = np.log(w)
wc, wm, wt, wd, wv = (L.HYG - L.IEI).diff(), L["^MOVE"].diff(), L.TLT.diff(), L["DX-Y.NYB"].diff(), L["^VIX"].diff()
ws = w["^GSPC"].to_numpy()
wmdd = np.array([ws[i:i + 14].min() / ws[i] - 1 if i + 13 < len(ws) else np.nan for i in range(len(ws))])
wA = ((wc < -0.005) & (wm > 0.10) & (wt < -0.01) & (wd > 0.005)).to_numpy()
wB = ((wc < -0.005) & (wm > 0.10) & (wt > 0.01) & (wv > 0.15)).to_numpy()
wu = wmdd[~np.isnan(wmdd)]
print(f"weekly unconditional P(MDD13w<-10%) {(wu < DD).mean():.1%} (complete windows only)")
for nm, s in [("A", wA), ("B", wB)]:
    allw = np.flatnonzero(s)
    ep = allw[np.r_[True, np.diff(allw) > 4]] if len(allw) else allw
    for lab, st in [("weekly bars, all firing weeks (v2 quote)", allw), ("weekly bars, de-clustered (gap 4w)", ep)]:
        m = wmdd[st]
        m = m[~np.isnan(m)]
        print(f"  {nm} {lab:42s} n={len(m):3d} P={(m < DD).mean():.0%} median MDD {np.median(m):+.1%}")
    st = episodes(sig_A() if nm == "A" else sig_B())
    print(f"  {nm} {'daily rolling 5d, de-clustered (gap 20bd)':42s} n={rate(st)[1]:3d} P={rate(st)[0]:.0%} median MDD {np.nanmedian(MDD[st]):+.1%}")

section("X2. Which leg carries the information? Drop one condition at a time (daily, de-clustered)")
for name, base, fn in [("A", base_A, sig_A), ("B", base_B, sig_B)]:
    rows = []
    for k in [None] + list(base):
        kw = dict(base)
        if k is not None:
            # remove the condition: '>= x' -> '>= -inf', '<= x' -> '<= +inf'
            kw[k] = -np.inf if k in ("mv", "dxy", "vix") or (name == "B" and k == "tlt") else np.inf
        st = episodes(fn(**kw))
        p, n = rate(st)
        rows.append([f"drop {k}" if k else "all four (v2)", len(st), f"{p:.0%}", f"{np.nanmedian(MDD[st]):+.1%}",
                     f"{np.median(DD252[st]):+.1%}"])
    print(f"\n{name}:")
    print(pd.DataFrame(rows, columns=["variant", "episodes", "P(MDD<-10%)", "med MDD", "med DD@fire"]).to_string(index=False))

section("X3. De-clustering gap sensitivity")
for g in [5, 10, 20, 40, 63]:
    ra, rb = episodes(SA, g), episodes(SB, g)
    print(f"gap {g:2d}bd: A n={rate(ra)[1]:3d} P={rate(ra)[0]:.0%} | B n={rate(rb)[1]:3d} P={rate(rb)[0]:.0%}")

section("X4. Early vs late firing: split episodes by S&P trend / drawdown at the firing date")
b200 = F["below200"].to_numpy()
for nm, st in [("A", EA), ("B", EBp), ("A or B", episodes(SA | SB))]:
    for lab, msk in [("S&P above 200d", ~b200[st]), ("S&P below 200d", b200[st]),
                     ("DD@fire > -5%", DD252[st] > -0.05), ("DD@fire <= -5%", DD252[st] <= -0.05)]:
        s2 = st[msk]
        p, n = rate(s2)
        print(f"  {nm:6s} {lab:16s} n={n:2d} P(MDD<-10%)={p if n else float('nan'):.0%} median MDD {np.nanmedian(MDD[s2]) if n else float('nan'):+.1%}")
e_ab = ELIG[~b200[ELIG]]
e_bb = ELIG[b200[ELIG]]
print(f"  uncond above 200d: P={(MDD[e_ab] < DD).mean():.0%} (n days {len(e_ab)}); below 200d: P={(MDD[e_bb] < DD).mean():.0%} (n days {len(e_bb)})")
print("  A/B episodes with DD@fire > -5% that were followed by MDD<-10%: " + ", ".join(
    f"{nm} {bd[s].date()} ({MDD[s]:+.1%})" for nm, st in [("A", EA), ("B", EBp)] for s in st
    if DD252[s] > -0.05 and not np.isnan(MDD[s]) and MDD[s] < DD))

section("X5. Forward-volatility view (the sizing target): median fwd 63d realised vol by state, and P(fwd vol > 20%)")
fv = FVOL[ELIG]
print(f"unconditional: median {np.median(fv):.1%}, P(>20%) {(fv > 0.20).mean():.0%}")
for k, s in {**{k: ACT[k] for k in ACT}, "STATE aggressive": state == "aggressive", "STATE neutral": state == "neutral",
             "STATE defensive": state == "defensive", "Hedge R corr>0.3": sbc > 0.3, "MOVE > 126": mlev > 126}.items():
    m = np.asarray(s, bool)[ELIG]
    print(f"  {k:28s} days {m.mean():5.1%}  median fwd vol {np.median(fv[m]):.1%}  P(fwd vol>20%) {(fv[m] > 0.20).mean():.0%}")


section("X6. Structural variants of the sizing rule (not grid-optimised); computable buckets only")
b = bucket.copy()
b["trend"] = np.where(b200, 2, 0)
ABv = AB.astype(int)


def make_state(cols, ab_mode):
    nr = (b[cols] == 2).sum(axis=1).to_numpy()
    ny = (b[cols] == 1).sum(axis=1).to_numpy()
    if ab_mode == "red":
        nr = nr + ABv
    dfn = (nr >= 2) | (AB if ab_mode == "defensive" else False)
    return np.where(dfn, "defensive", np.where((nr >= 1) | (ny >= 2), "neutral", "aggressive"))


VARIANTS = {
    "C0 v2 as written": (["ratevol", "eqvol", "hedge", "fund", "diag"], "defensive"),
    "C1 drop hedge bucket": (["ratevol", "eqvol", "fund", "diag"], "defensive"),
    "C2 drop hedge, A|B = one red": (["ratevol", "eqvol", "fund", "diag"], "red"),
    "C3 C2 + trend red (S&P<200d)": (["ratevol", "eqvol", "fund", "diag", "trend"], "red"),
    "C4 C3 without composite": (["ratevol", "eqvol", "fund", "diag", "trend"], "none"),
}
EXPO = {"aggressive": 1.0, "neutral": 0.7, "defensive": 0.4}
rspx = np.r_[np.nan, np.diff(spx) / spx[:-1]]
rows = []
for nm, (cols, mode) in VARIANTS.items():
    st_ = make_state(cols, mode)
    for per, (lo, hi) in {"2008-26": (s0, N), "2008-16": (s0, split_pos), "2017-26": (split_pos, N)}.items():
        e_ = np.arange(lo, hi)
        e_ = e_[~np.isnan(MDD[e_])]
        cells = []
        for lab in ["aggressive", "neutral", "defensive"]:
            m = st_[e_] == lab
            cells.append(f"{m.mean():4.0%} {(MDD[e_][m] < DD).mean() if m.any() else np.nan:4.0%} {np.median(FVOL[e_][m]) if m.any() else np.nan:5.1%}")
        rows.append([nm, per] + cells)
print("cells = share of days / P(S&P MDD63<-10%) / median fwd 63d vol")
print(pd.DataFrame(rows, columns=["variant", "period", "aggressive", "neutral", "defensive"]).to_string(index=False))


def perf(expo, lo, hi):
    rr = (expo[lo - 1:hi - 1] * rspx[lo:hi])
    rr = rr[~np.isnan(rr)]
    eq = np.cumprod(1 + rr)
    mdd = (eq / np.maximum.accumulate(eq) - 1).min()
    w63 = (eq[63:] / eq[:-63] - 1).min()
    return rr.mean() * 252, rr.std() * np.sqrt(252), mdd, w63


print("\nExposure backtest of S&P price index, 2008-01..2026-10, exposure decided at close t applied to t+1 (cash = 0%),"
      " 100/70/40% by state, vs a constant exposure equal to the variant's average exposure")
rows = []
for nm, (cols, mode) in VARIANTS.items():
    st_ = make_state(cols, mode)
    ex = np.vectorize(EXPO.get)(st_).astype(float)
    avg = ex[s0:].mean()
    a1 = perf(ex, s0, N)
    a2 = perf(np.full(N, avg), s0, N)
    rows.append([nm, f"{avg:.0%}", f"{a1[0]:+.1%}/{a1[1]:.1%}", f"{a1[0] / a1[1]:.2f}", f"{a1[2]:+.1%}", f"{a1[3]:+.1%}",
                 f"{a2[0]:+.1%}/{a2[1]:.1%}", f"{a2[0] / a2[1]:.2f}", f"{a2[2]:+.1%}", f"{a2[3]:+.1%}",
                 f"{(np.abs(np.diff(ex[s0:])) > 0).sum() / ((N - s0) / 252):.0f}"])
print(pd.DataFrame(rows, columns=["variant", "avg expo", "ret/vol", "ret/vol ratio", "MaxDD", "worst 63d",
                                  "const: ret/vol", "const ratio", "const MaxDD", "const worst 63d",
                                  "changes/yr"]).to_string(index=False))

print("\nState by variant, recent days (credit bucket unavailable -> treated as green):")
cur2 = pd.DataFrame({nm: make_state(cols, mode) for nm, (cols, mode) in VARIANTS.items()}, index=bd)
cur2["S&P<200d"] = b200
print(cur2.loc["2026-09-22":].to_string())
