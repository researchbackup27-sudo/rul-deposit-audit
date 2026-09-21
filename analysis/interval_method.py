"""
nb16: does the verdict depend on HOW the interval is built, and on not adjusting for
the per-channel scan?

Two objections are answered here with recomputes over the checkpointed per-unit arrays.
No signal is touched and nothing is retrained.

  (a) The audit uses a percentile bootstrap, which is known to under-cover on a log-log
      slope at small n. Two alternatives are scored on the SAME bootstrap draws: a BCa
      interval (bias-corrected and accelerated, jackknife acceleration) and the ordinary
      OLS t interval, which makes entirely different assumptions. Coverage is measured
      against channels of known exponent, then every panel verdict is recomputed under
      each method.

  (b) Equations eq:informative and eq:clock are applied independently to every channel a
      deposit ships and produce no p-value to adjust. A Bonferroni-equivalent is available
      anyway: widen each interval to level 1 - 0.05/m, m being the channels scored in that
      deposit, so the FAMILY-wise error over the scan is controlled at 0.05. The panel is
      rescored under that too.

CPU only.
"""
import json, os, sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import norm, t as tdist
sys.path.insert(0, ".")
import rul_audit as R

OUT = Path("interval_method"); OUT.mkdir(exist_ok=True)
# Not redistributed; see the README.
ACC = Path(os.environ.get("RUL_ACCUMULATORS", "accumulators"))
DEPOSITS = ["polymer", "cmapss_raw", "cmapss_norm", "femto", "ims"]
SEEDS = list(range(1, 21))
cfg = R.AuditConfig()


def head(t): print("\n" + "=" * 78); print(t); print("=" * 78)


def slopes_and_point(d, L, seed, n_boot):
    """The library's own fit, plus the raw bootstrap slope draws behind it."""
    d = np.abs(np.asarray(d, float)); L = np.asarray(L, float)
    ok = np.isfinite(d) & np.isfinite(L) & (d > 0) & (L > 0)
    d, L = d[ok], L[ok]
    if len(d) < cfg.min_units_for_beta:
        return None
    x, y = np.log(L), np.log(d)
    if np.ptp(x) == 0:
        return None
    order = np.lexsort((y, x)); x, y = x[order], y[order]
    beta, const = np.polyfit(x, y, 1)
    rng = np.random.default_rng(seed); m = len(x); out = []
    for start in range(0, n_boot, 2000):
        k = min(2000, n_boot - start)
        idx = rng.integers(0, m, size=(k, m))
        X, Y = x[idx], y[idx]
        xd = X - X.mean(axis=1, keepdims=True)
        yd = Y - Y.mean(axis=1, keepdims=True)
        den = (xd * xd).sum(axis=1); good = den > 0
        out.append((xd * yd).sum(axis=1)[good] / den[good])
    s = np.concatenate(out)
    return (beta, s, x, y) if s.size else None


def three_intervals(d, L, seed, n_boot=None, alpha=0.05):
    """percentile / BCa / OLS-t, all from one fit and one set of draws."""
    got = slopes_and_point(d, L, seed, n_boot or cfg.n_boot)
    if got is None:
        return None
    beta, s, x, y = got
    n = len(x)
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out = {"beta": float(beta), "n": n,
           "percentile": (float(np.percentile(s, lo_q)), float(np.percentile(s, hi_q)))}

    # --- BCa: bias correction from the draws, acceleration from a jackknife over units
    prop = float(np.mean(s < beta))
    prop = min(max(prop, 1.0 / (len(s) + 1)), 1.0 - 1.0 / (len(s) + 1))
    z0 = norm.ppf(prop)
    jack = np.empty(n)
    for i in range(n):
        xi = np.delete(x, i); yi = np.delete(y, i)
        jack[i] = np.polyfit(xi, yi, 1)[0] if np.ptp(xi) > 0 else np.nan
    if np.isfinite(jack).all():
        dev = jack.mean() - jack
        den = 6.0 * (np.sum(dev ** 2) ** 1.5)
        a = float(np.sum(dev ** 3) / den) if den > 0 else 0.0
    else:
        a = 0.0
    zl, zu = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)
    def adj(z):
        denom = 1 - a * (z0 + z)
        return norm.cdf(z0 + (z0 + z) / denom) if denom != 0 else norm.cdf(z0 + z)
    a1, a2 = adj(zl), adj(zu)
    out["bca"] = (float(np.percentile(s, 100 * a1)), float(np.percentile(s, 100 * a2)))

    # --- ordinary OLS t interval on the slope
    pred = beta * x + np.polyfit(x, y, 1)[1]
    dof = n - 2
    if dof > 0:
        sxx = float(np.sum((x - x.mean()) ** 2))
        sigma2 = float(np.sum((y - pred) ** 2) / dof)
        se = np.sqrt(sigma2 / sxx) if sxx > 0 else np.nan
        tcrit = tdist.ppf(1 - alpha / 2, dof)
        out["ols_t"] = (float(beta - tcrit * se), float(beta + tcrit * se))
    else:
        out["ols_t"] = (np.nan, np.nan)
    return out


# ============================================================ (a) coverage of each method
head("(a) COVERAGE against channels of known exponent, same draws for all three methods")
NS = [5, 6, 8, 13, 17, 60]
BETAS = [0.0, 0.5, 1.0]
TRIALS, NBOOT = 1000, 2000
METHODS = ["percentile", "bca", "ols_t"]
rng = np.random.default_rng(20260921)

# The OLS-t interval assumes homoscedastic, normal residuals in log space. Scoring it only
# under a generator that satisfies that assumption would rig the comparison in its favour,
# so coverage is measured under two misspecifications as well.
def resid(kind, n, x, rng):
    if kind == "lognormal":                       # the t interval's own model
        return rng.normal(0, 0.5, n)
    if kind == "heteroscedastic":                 # spread grows with lifetime
        return rng.normal(0, 0.5, n) * (0.4 + 1.2 * (x - x.min()) / max(np.ptp(x), 1e-9))
    if kind == "heavy_tailed":                    # t_3 residuals, occasional large outliers
        return 0.5 * rng.standard_t(3, n)
    raise ValueError(kind)

REGIMES = ["lognormal", "heteroscedastic", "heavy_tailed"]
print(f"  {TRIALS} trials per cell, n_boot = {NBOOT}, nominal 0.95")
cover = {}
for reg in REGIMES:
    print(f"\n  residuals: {reg}")
    print(f"{'n':>4s}" + "".join(f"{m:>13s}" for m in METHODS))
    cover[reg] = {}
    for n in NS:
        hits = {m: 0 for m in METHODS}; tot = 0
        for _ in range(TRIALS):
            tb = BETAS[int(rng.integers(len(BETAS)))]
            L = np.exp(rng.uniform(np.log(100), np.log(1600), n))
            d = np.exp(tb * np.log(L) + resid(reg, n, np.log(L), rng))
            r = three_intervals(d, L, int(rng.integers(1 << 62)), n_boot=NBOOT)
            if r is None:
                continue
            tot += 1
            for m in METHODS:
                lo, hi = r[m]
                if np.isfinite(lo) and lo <= tb <= hi:
                    hits[m] += 1
        cover[reg][n] = {m: hits[m] / tot for m in METHODS}
        print(f"{n:4d}" + "".join(f"{cover[reg][n][m]:13.3f}" for m in METHODS))
    worst = {m: min(cover[reg][n][m] for n in NS) for m in METHODS}
    print("  worst cell: " + ", ".join(f"{m} {worst[m]:.3f}" for m in METHODS))

# ============================================================ (b) rescore the whole panel
head("(b) EVERY PANEL VERDICT under percentile / BCa / OLS-t, and Bonferroni-widened")
if not ACC.is_dir():
    print(f"  SKIPPED: no per-unit arrays at {ACC}/. Section (a) is simulation only.")
    json.dump({"coverage": {r: {str(k): v for k, v in cover[r].items()}
                            for r in REGIMES},
               "coverage_trials": TRIALS, "coverage_n_boot": NBOOT,
               "coverage_regimes": REGIMES, "clearance_counts": None,
               "headline_holds": None, "methods": METHODS},
              open(OUT / "interval_method.json", "w"), indent=2)
    sys.exit(0)
def load(name):
    dd = json.load(open(ACC / f"{name}.json"))
    return dd["acc"], dd["meta"].get("label_channel")

rows = []
for name in DEPOSITS:
    acc, lab = load(name)
    chans = sorted(acc)
    m_scan = len(chans)
    for ch in chans:
        a = acc[ch]
        for seed in SEEDS:
            sd = R._seed_for(seed, f"beta|{ch}")
            base = three_intervals(a["delta"], a["life"], sd)
            if base is None:
                continue
            # two-sided level 2*0.05/m == one-sided 0.05/m over the scan; see the docstring
            wide = three_intervals(a["delta"], a["life"], sd, alpha=2.0 * 0.05 / m_scan)
            rec = {"deposit": name, "channel": ch, "seed": seed, "beta": base["beta"],
                   "n": base["n"], "m_scan": m_scan}
            for meth in METHODS:
                lo, hi = base[meth]
                rec[f"cleared_{meth}"] = bool(np.isfinite(hi) and hi < 1.0)
                rec[f"lo_{meth}"], rec[f"hi_{meth}"] = lo, hi
            lo, hi = wide["percentile"]
            rec["cleared_bonferroni"] = bool(np.isfinite(hi) and hi < 1.0)
            rec["lo_bonferroni"], rec["hi_bonferroni"] = lo, hi
            rows.append(rec)
df = pd.DataFrame(rows)
df.to_csv(OUT / "per_channel_intervals.csv", index=False)

ARMS = METHODS + ["bonferroni"]
LABELS = {n: load(n)[1] for n in DEPOSITS}

# The interval method changes only the CLEARANCE component of Equation eq:indicator. The
# other three conditions (informative, independent, not the target channel) are unchanged,
# so the paper's own verdict table supplies them and only clearance is swapped. That makes
# the counts below directly comparable to the indicator column of Table tab:l3.
base_rows = []
for name in DEPOSITS:
    acc, lab = load(name)
    accm = dict(acc); accm[R.META_KEY] = {"label_channel": lab}
    for seed in SEEDS:
        t = R.l3_verdicts(accm, seed, cfg=cfg)
        t.insert(0, "deposit", name); t["seed"] = seed
        base_rows.append(t)
base = pd.concat(base_rows, ignore_index=True)
key = ["deposit", "channel", "seed"]
merged = base.merge(df, on=key, how="left", suffixes=("", "_iv"))
# reconstruct the composite exactly as the library does, swapping only clearance
for arm in ARMS:
    merged[f"indicator_{arm}"] = (
        merged["informative"].fillna(False).astype(bool)
        & merged[f"cleared_{arm}"].fillna(False).astype(bool)
        & merged["beta_fitted"].fillna(False).astype(bool)
        & merged["independent"].fillna(False).astype(bool)
        & ~merged["channel"].eq(merged["deposit"].map(LABELS)))

print("\n  INDICATOR counts per deposit (min-max over the 20 seeds), Equation eq:indicator")
print("  with only the clearance component swapped\n")
print(f"{'deposit':>14s}{'paper':>9s}" + "".join(f"{a:>14s}" for a in ARMS))
counts = {}
for name in DEPOSITS:
    g = merged[merged.deposit == name]
    paper = g.groupby("seed")["indicator"].sum()
    pl, ph = int(paper.min()), int(paper.max())
    cells = []
    counts[name] = {"paper": [pl, ph]}
    for arm in ARMS:
        ps = g.groupby("seed")[f"indicator_{arm}"].sum()
        lo_c, hi_c = int(ps.min()), int(ps.max())
        counts[name][arm] = [lo_c, hi_c]
        cells.append(f"{lo_c}" if lo_c == hi_c else f"{lo_c}-{hi_c}")
    ptxt = f"{pl}" if pl == ph else f"{pl}-{ph}"
    print(f"{name:>14s}{ptxt:>9s}" + "".join(f"{c:>14s}" for c in cells))
merged.to_csv(OUT / "indicator_counts_by_method.csv", index=False)

head("HEADLINE VERDICTS UNDER EVERY INTERVAL METHOD")
HEAD = [("femto", "acc_horiz_kurt", "must NOT clear"),
        ("femto", "temp_rtd",       "must clear"),
        ("polymer", "motor_amp",    "must clear"),
        ("cmapss_norm", "s9",       "must NOT clear")]
ok_all = True
for dep, ch, want in HEAD:
    g = df[(df.deposit == dep) & (df.channel == ch)]
    if not len(g):
        print(f"  {dep}/{ch}: ABSENT"); ok_all = False; continue
    bits = []
    for arm in ARMS:
        frac = g[f"cleared_{arm}"].mean()
        bits.append(f"{arm} {frac:.0%}")
    beta = g.beta.iloc[0]
    holds = (all(g[f"cleared_{a}"].eq(False).all() for a in ARMS) if "NOT" in want
             else all(g[f"cleared_{a}"].eq(True).all() for a in ARMS))
    ok_all &= holds
    print(f"  {dep:12s}/{ch:16s} beta={beta:7.3f}  {want:15s} "
          f"[{'HOLDS' if holds else 'BREAKS'}]  cleared: " + ", ".join(bits))

art = {"coverage": {r: {str(k): v for k, v in cover[r].items()} for r in REGIMES},
       "coverage_trials": TRIALS, "coverage_n_boot": NBOOT, "coverage_regimes": REGIMES,
       "clearance_counts": counts, "headline_holds": bool(ok_all),
       "methods": ARMS}
json.dump(art, open(OUT / "interval_method.json", "w"), indent=2)
print(f"\nartifacts -> {OUT/'interval_method.json'}, {OUT/'per_channel_intervals.csv'}")
print("\nHEADLINE VERDICTS HOLD UNDER ALL FOUR" if ok_all else "\nA HEADLINE VERDICT MOVES")
sys.exit(0 if ok_all else 1)
