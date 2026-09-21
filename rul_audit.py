"""
rul_audit: a three-leg reusability audit for run-to-failure prognostics deposits.

The audit scores a DATASET, not a model. Nothing is trained.

    Leg 1  Is the released target recoverable from the record length alone?
    Leg 2  Was the failure event established independently of the channels? (documentary)
    Leg 3  Does an informative, independent, non-clock channel exist?

Leg 3 is where the scaling exponent lives. Trendability rewards any quantity that moves
steadily through a record, and elapsed time moves steadily through every record, so a clock
passes it. Fitting |delta| ~ lifetime**beta separates the two: a channel that accumulates
with elapsed time gives beta -> 1, a channel that reaches a comparable end state however
long the unit lived gives beta -> 0. The null is beta = 1, and clearance is ONE-SIDED: a
channel is cleared only when its whole interval lies BELOW unity. An excursion that grows
faster than the lifetime (beta > 1) is more strongly time-driven than a clock, not less, so
it is not evidence of damage information.

Design note. `l3_accumulate` is expensive and deterministic; `l3_verdicts` is cheap and
seed-dependent. Keeping them apart means a seed sweep, a changed threshold, or a new
criterion costs no access to the raw signals. Checkpoint the accumulator and re-use it.

Requires numpy, pandas, scipy. CPU only.

Run `python rul_audit.py` to execute the positive control, which needs no data.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "AuditConfig", "leg1", "icc21", "scaling_exponent", "prognosability", "monotonicity",
    "l3_accumulate", "l3_verdicts", "seed_sweep", "positive_control",
]


@dataclass(frozen=True)
class AuditConfig:
    """Thresholds and resampling settings. Fix these before scoring anything."""
    l1_fail_r2: float = 0.90      # at or above, the target is recoverable from the record
    rho_min: float = 0.30         # trendability cut; the CI lower bound must clear it
    redundant: float = 0.80       # |rho| against the target-defining channel
    beta_null: float = 1.0        # a pure clock scales as lifetime ** 1
    min_units_for_beta: int = 5   # below this no interval is produced
    burn_in: float = 0.10         # leading fraction of each record discarded
    n_boot: int = 10_000


def _seed_for(master: int, key: str) -> int:
    """Per-(seed, channel) stream that is reproducible across processes.

    Deliberately not `hash(key)`: Python randomises string hashing per process unless
    PYTHONHASHSEED is pinned, which silently makes intervals irreproducible between runs.
    """
    digest = hashlib.md5(f"{master}|{key}".encode(), usedforsecurity=False).hexdigest()
    return int(digest[:8], 16) % (2 ** 31)


def _boot_ci(values, fn, seed: int, n: int) -> tuple[float, float]:
    v = np.asarray(values, float)
    v = np.sort(v[np.isfinite(v)])   # sorted so the interval does not depend on unit order
    if len(v) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = fn(v[rng.integers(0, len(v), size=(n, len(v)))], axis=1)
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def icc21(a, b) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single measurement.

    Correlation says two quantities move together; agreement says they are the same
    number. A target that is the record length in different units correlates perfectly
    and agrees not at all, so Leg 1 reports both. Shrout and Fleiss (1979), case 2.
    """
    m = np.column_stack([np.asarray(a, float), np.asarray(b, float)])
    m = m[np.isfinite(m).all(axis=1)]
    n, k = m.shape
    if n < 2:
        return float("nan")
    grand = m.mean()
    # between-subject, between-rater and residual mean squares
    msr = k * ((m.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    msc = n * ((m.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    sse = ((m - m.mean(axis=0)[None, :] - m.mean(axis=1)[:, None] + grand) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / den) if den != 0 else float("nan")


# --------------------------------------------------------------------------- Leg 1
def leg1(targets: Sequence[float], lengths: Sequence[float],
         cfg: AuditConfig = AuditConfig()) -> dict:
    """Is the released target recoverable from the record length?

    `targets` is one released target per unit, `lengths` the corresponding record length.
    The OLS fit is scored IN SAMPLE on purpose: that is generous to the deposit, since a
    penalised score would bias the audit toward finding leakage.

    Splits whose units run to failure return r2 == 1 by construction; no interval is placed
    on such a value and `by_construction` is set. That flag keys on r2, not on exact
    identity, because a target defined as N * dt is equally a construction.

    `zero_param_mae` and `zero_param_mape_pct` compare the record length directly against
    the target, so they are meaningful only when the two share units. Where they do not,
    read `slope`: a value of dt means the target is the record length in other units.
    """
    L = np.asarray(targets, float)
    N = np.asarray(lengths, float)
    ok = np.isfinite(L) & np.isfinite(N)
    L, N = L[ok], N[ok]
    u = len(L)
    if u < 3:
        raise ValueError(f"leg1 needs at least 3 units, got {u}")

    slope, intercept = np.polyfit(N, L, 1)
    pred = slope * N + intercept
    ss_res = float(np.sum((L - pred) ** 2))
    ss_tot = float(np.sum((L - L.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    # A target identical to the record length up to float round-off is still an identity;
    # testing with == under-counts any target that passed through a unit conversion.
    tol = 1e-9 * max(1.0, float(np.max(np.abs(N))) if len(N) else 1.0)
    exact = int(np.sum(np.isclose(L, N, rtol=0.0, atol=tol)))

    # The zero-parameter error is only interpretable when the target and the record share
    # units. Where the target is a scaled record length (N * dt), read `slope` instead.
    nz = L != 0
    mape = (float(np.mean(np.abs(N[nz] - L[nz]) / np.abs(L[nz])) * 100)
            if nz.any() else np.nan)

    out = {
        "n_units": u,
        "r2": float(r2),
        "r2_adjusted": float(1 - (1 - r2) * (u - 1) / (u - 2)) if u > 2 else np.nan,
        "r2_null_expectation": 1.0 / (u - 1),   # E[R2] for one predictor, no relationship
        "slope": float(slope),
        "intercept": float(intercept),
        "exact_identity_n": exact,
        "exact_identity_rate": exact / u,
        # correlation establishes covariation; ICC asks for agreement, which is the
        # stronger claim and the one that matters when a target may BE the record length
        "icc21": icc21(L, N),
        "spearman": float(stats.spearmanr(N, L).statistic),
        "pearson": float(stats.pearsonr(N, L).statistic),
        "zero_param_mae": float(np.mean(np.abs(N - L))),
        "zero_param_mape_pct": mape,
        "zero_target_units": int((~nz).sum()),
        # A target that is any affine function of the record length is a construction,
        # not a measurement. Keying this on exact identity would miss L = N * dt.
        # rtol is pinned to 0: np.isclose defaults to rtol=1e-5, which would make the
        # stated 1e-9 rule four orders of magnitude looser than it reads.
        "by_construction": (bool(np.isclose(r2, 1.0, rtol=0.0, atol=1e-9))
                            if ss_tot > 0 else None),
        # A constant target has no variance to explain, so r2 is undefined. Reporting that
        # as PASS would read a NaN as evidence the target is not recoverable.
        "verdict": ("NOT FITTED" if ss_tot <= 0 else
                    "FAIL" if r2 >= cfg.l1_fail_r2 else "PASS"),
    }
    return out


# --------------------------------------------------------------------------- Leg 3 parts
def scaling_exponent(deltas: Sequence[float], lifetimes: Sequence[float], seed: int,
                     cfg: AuditConfig = AuditConfig()) -> dict:
    """Fit |delta| = kappa * lifetime ** beta and bootstrap beta.

    beta -> 1 is a clock, beta -> 0 a damage indicator. The POINT estimate is deterministic;
    only the interval depends on the seed. Returns beta = nan when fewer than
    `min_units_for_beta` units survive, which must be reported as 'not fitted' and never as
    'is a clock': an absent fit is not evidence of anything.
    """
    d = np.abs(np.asarray(deltas, float))
    L = np.asarray(lifetimes, float)
    ok = np.isfinite(d) & np.isfinite(L) & (d > 0) & (L > 0)
    d, L = d[ok], L[ok]
    unfitted = {"n": int(len(d)), "beta": np.nan, "ci": (np.nan, np.nan),
                "fit_r2": np.nan, "fitted": False, "n_boot_effective": 0}
    if len(d) < cfg.min_units_for_beta:
        return unfitted

    x, y = np.log(L), np.log(d)
    # No spread in log-lifetime means no slope is identified: every unit lived the same
    # length, so nothing can be learned about how the excursion scales with lifetime.
    # Without this guard np.polyfit returns a meaningless slope and every bootstrap draw
    # is degenerate, which used to raise IndexError from an empty percentile.
    if np.ptp(x) == 0:
        return unfitted
    # Sorted so that the same deposit loaded in a different unit order gives the same
    # interval; the point estimate is order-invariant already.
    order = np.lexsort((y, x))
    x, y = x[order], y[order]
    beta, const = np.polyfit(x, y, 1)
    pred = beta * x + const
    fit_r2 = (1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
              if np.var(y) > 0 else np.nan)

    # closed-form OLS slope, vectorised over bootstrap draws
    rng = np.random.default_rng(seed)
    m = len(x)
    slopes = []
    for start in range(0, cfg.n_boot, 2000):
        k = min(2000, cfg.n_boot - start)
        idx = rng.integers(0, m, size=(k, m))
        X, Y = x[idx], y[idx]
        xd = X - X.mean(axis=1, keepdims=True)
        yd = Y - Y.mean(axis=1, keepdims=True)
        den = (xd * xd).sum(axis=1)
        good = den > 0
        slopes.append((xd * yd).sum(axis=1)[good] / den[good])
    slopes = np.concatenate(slopes)
    # A resample holding only one distinct lifetime identifies no slope and is dropped, so
    # the interval is conditional on the surviving draws; n_boot_effective reports how many
    # that was. If none survive there is no interval to report.
    if slopes.size == 0:
        return unfitted
    return {"n": int(len(d)), "beta": float(beta),
            "ci": (float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))),
            "fit_r2": float(fit_r2), "fitted": True,
            "n_boot_effective": int(slopes.size)}


def prognosability(end_values, start_values) -> float:
    """Coble and Hines form: how tightly units converge to a common end-of-life value."""
    e, s = np.asarray(end_values, float), np.asarray(start_values, float)
    ok = np.isfinite(e) & np.isfinite(s)
    e, s = e[ok], s[ok]
    if len(e) < 3:
        return np.nan
    denom = float(np.mean(np.abs(e - s)))
    return float(np.exp(-np.std(e) / denom)) if denom > 0 else np.nan


def monotonicity(series) -> float:
    """|fraction rising - fraction falling| over one unit. 1 is perfectly monotone."""
    d = np.diff(np.asarray(series, float))
    d = d[np.isfinite(d) & (d != 0)]
    return float(abs((d > 0).mean() - (d < 0).mean())) if len(d) else np.nan


#: reserved key under which `l3_accumulate` stores what the verdict step needs to know.
META_KEY = "__meta__"
_UNSET = object()

#: How the per-unit excursion is measured. "endpoint" is the definition the audit is
#: specified against; the other two exist so the verdicts can be re-scored under a
#: different reading of the same records without touching the raw signals.
DELTA_MODES = ("endpoint", "smoothed", "raw")


def _deltas(a: Mapping[str, list], mode: str) -> list:
    if mode not in DELTA_MODES:
        raise ValueError(f"delta_mode must be one of {DELTA_MODES}, got {mode!r}")
    key = {"endpoint": "delta", "smoothed": "delta_smoothed", "raw": "delta_raw"}[mode]
    if key not in a:
        # Accumulators checkpointed before the alternatives were stored carry only the
        # endpoint excursion; "smoothed" is still recoverable from the retained ends.
        if mode == "smoothed" and "end" in a and "start" in a:
            return list(np.asarray(a["end"], float) - np.asarray(a["start"], float))
        raise KeyError(
            f"this accumulator has no {key!r}; it predates delta_mode={mode!r} and the "
            f"deposit must be re-accumulated to score under it")
    return a[key]


def _trim(frame: pd.DataFrame, frac: float) -> pd.DataFrame:
    return frame.iloc[int(round(frac * len(frame))):].reset_index(drop=True)


def l3_accumulate(units: Mapping[object, pd.DataFrame],
                  lifetimes: Mapping[object, float],
                  label_channel: str | None = None,
                  cfg: AuditConfig = AuditConfig()) -> Dict[str, dict]:
    """EXPENSIVE and DETERMINISTIC. No seed is used here.

    `units` maps a unit id to a DataFrame whose columns are channels and whose rows are
    ordered samples. ROW ORDER IS USED POSITIONALLY and is never inferred: progress through
    a record is row index over row count, so a frame that is not already in chronological
    order produces a plausible-looking but meaningless trend. An index is not consulted for
    ordering; sort before calling. A datetime index is checked and refused when it is not
    monotonic, which catches the common case only.
    `lifetimes` maps the same ids to the unit's lifetime in native units.
    `label_channel` is the channel the deposit says the event was timed from, or None.

    Returns per-channel arrays with one entry per unit that the channel was scorable in,
    plus a reserved `__meta__` entry recording the label channel and how many units were
    skipped. Raises if `label_channel` matches no column or any unit has no lifetime, both
    of which otherwise produce a clean-looking but empty result. Checkpoint this: every
    seed afterwards is a cheap recompute over these numbers.
    """
    names = sorted({c for f in units.values() for c in f.columns})
    if META_KEY in names:
        raise ValueError(f"{META_KEY!r} is reserved and cannot be a channel name")
    # A label channel that matches no column silently turns the independence criterion into
    # a pass-through: nothing is compared, every channel reports redundancy NaN, and an
    # exact copy of the target-defining channel is certified independent. One typo is
    # enough, so it is an error rather than a warning.
    if label_channel is not None:
        seen = sum(1 for f in units.values() if label_channel in f.columns)
        if seen == 0:
            raise ValueError(
                f"label_channel {label_channel!r} matches no column in any unit; "
                f"available channels: {names}")
    # A lifetime that is missing rather than absent scores the whole deposit as having no
    # indicator channel, and the output is indistinguishable from a real result. The usual
    # cause is a key-type mismatch between `units` and `lifetimes` (int vs str ids).
    missing = [u for u in units if u not in lifetimes]
    if missing:
        raise ValueError(
            f"{len(missing)} unit(s) have no entry in `lifetimes`, e.g. {missing[:5]}; "
            f"unit id types are {sorted({type(u).__name__ for u in units})} and lifetime "
            f"id types are {sorted({type(u).__name__ for u in lifetimes})}")

    acc = {c: {"rho": [], "delta": [], "delta_smoothed": [], "delta_raw": [], "life": [],
               "mono": [], "start": [], "end": [], "red": [], "frac_finite": []}
           for c in names}
    skipped = []

    for uid, full in units.items():
        # Only catches a datetime index. Positional order is the contract either way.
        if isinstance(full.index, pd.DatetimeIndex) and not full.index.is_monotonic_increasing:
            raise ValueError(
                f"unit {uid!r} has a datetime index that is not increasing; rows are scored "
                f"in positional order, so sort the frame before calling")
        f = _trim(full, cfg.burn_in)
        if len(f) < 10:
            skipped.append(uid)          # too short after burn-in to score
            continue
        progress = np.arange(len(f)) / (len(f) - 1)
        lab = (f[label_channel].to_numpy(float)
               if label_channel and label_channel in f else None)
        for c in names:
            if c not in f:
                continue
            try:
                v = f[c].to_numpy(float)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"channel {c!r} in unit {uid!r} is not numeric") from exc
            ok = np.isfinite(v)
            if ok.sum() < 10 or np.std(v[ok]) == 0:
                continue      # channel is absent or constant in this unit
            a = acc[c]
            a["rho"].append(float(abs(stats.spearmanr(progress[ok], v[ok]).statistic)))
            k = max(3, len(v) // 20)
            sm = pd.Series(v).rolling(k, min_periods=1).mean().to_numpy()
            # every reduction below can meet an all-NaN window on a gappy channel; that is
            # reported through frac_finite rather than as a stream of RuntimeWarnings
            warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
            start_v, end_v = float(np.nanmean(sm[:k])), float(np.nanmean(sm[-k:]))
            a["frac_finite"].append(float(ok.sum() / len(v)))
            a["start"].append(start_v)
            a["end"].append(end_v)
            # Three ways to measure the excursion, all stored so that scoring under any of
            # them costs no re-accumulation. They differ only in how the two ends are
            # estimated, and that difference is not neutral:
            #   endpoint   last value of the smoothed series minus its first. The first
            #              value of a min_periods=1 rolling mean is a single raw sample, so
            #              the terminal state is well estimated and the initial one is not.
            #   smoothed   mean of the first and last k values of the smoothed series. The
            #              start is cleaner, but the end now spans 2k-1 raw samples and k
            #              grows with the record, so long units have their terminal value
            #              averaged away more than short ones.
            #   raw        mean of the first and last k RAW samples. Equal windows at both
            #              ends, which is the only one of the three that is symmetric.
            a["delta"].append(float(sm[-1] - sm[0]))
            a["delta_smoothed"].append(float(np.nanmean(sm[-k:]) - np.nanmean(sm[:k])))
            a["delta_raw"].append(float(np.nanmean(v[-k:]) - np.nanmean(v[:k])))
            a["life"].append(float(lifetimes.get(uid, np.nan)))
            a["mono"].append(monotonicity(sm))
            if lab is not None and c != label_channel:
                both = ok & np.isfinite(lab)
                if both.sum() > 10 and np.std(lab[both]) > 0:
                    a["red"].append(float(abs(stats.spearmanr(v[both], lab[both]).statistic)))

    # A channel scored on far fewer units than its peers is usually a naming mismatch
    # between units rather than a real absence, and it produces a verdict from a subset
    # without saying so. Warn rather than raise: partial coverage is sometimes genuine.
    counts = {c: len(a["rho"]) for c, a in acc.items() if a["rho"]}
    if len(counts) > 2:
        typical = float(np.median(list(counts.values())))
        thin = {c: n for c, n in counts.items() if n < 0.5 * typical}
        if thin:
            warnings.warn(
                f"these channels were scored on far fewer units than the median of "
                f"{typical:.0f}, which usually means a channel name differs between "
                f"units: {thin}", stacklevel=2)

    # Carried with the accumulator so the verdict step cannot be handed a different label
    # channel than the one the excursions were built against.
    acc[META_KEY] = {"label_channel": label_channel,
                     "n_units_total": len(units),
                     "n_units_scored": len(units) - len(skipped),
                     "skipped_units": skipped}
    return acc


def l3_verdicts(acc: Mapping[str, dict], seed: int = 1,
                label_channel=_UNSET, delta_mode: str = "endpoint",
                cfg: AuditConfig = AuditConfig()) -> pd.DataFrame:
    """CHEAP and SEED-DEPENDENT. Turns an accumulator into one row per channel.

    The label channel is taken from the accumulator's `__meta__` entry. Passing one
    explicitly is allowed only when it agrees with what the accumulator was built against.

    `is_clock` is None when the exponent could not be fitted. That is deliberate: treating
    an absent fit as a clock fails closed and prints a confident verdict from no evidence.

    `delta_mode` selects how the excursion is measured; see DELTA_MODES. "endpoint" is the
    specified definition and the other two are for sensitivity, not for picking a result.

    Clearance is ONE-SIDED. `is_clock` says the interval covers unity, `superlinear` says it
    lies entirely above it, and a channel is certified an indicator only when neither holds,
    i.e. only when the whole interval sits below unity. An excursion that grows faster than
    the lifetime is more strongly driven by elapsed time than a clock is, so it carries no
    damage information and must not be cleared by a two-sided test.
    """
    meta = dict(acc.get(META_KEY) or {})
    if label_channel is _UNSET:
        lab = meta.get("label_channel")
    else:
        lab = label_channel
        if META_KEY in acc and meta.get("label_channel") != lab:
            raise ValueError(
                f"label_channel {lab!r} was passed but the accumulator was built against "
                f"{meta.get('label_channel')!r}; redundancy was measured against the latter")
    rows = []
    for c in sorted(k for k in acc if k != META_KEY):
        a = acc[c]
        rho = np.asarray(a["rho"], float)
        n_valid = int(np.isfinite(rho).sum())
        if n_valid < 3:
            rows.append({"channel": c, "delta_mode": delta_mode,
                         "n_units": len(a["rho"]), "n_valid": n_valid,
                         "trend": np.nan, "trend_ci_lo": np.nan, "beta": np.nan,
                         "beta_ci_lo": np.nan, "beta_ci_hi": np.nan,
                         "beta_fit_r2": np.nan,
                         "prognosability": np.nan, "monotonicity": np.nan,
                         "redundancy": np.nan, "informative": False, "is_clock": None,
                         "superlinear": None, "frac_finite": np.nan, "beta_fitted": False,
                         "beta_covers_both": False, "beta_n_units": 0,
                         "n_boot_effective": 0, "insufficient_data": True,
                         "independent": True, "indicator": False})
            continue

        lo, _ = _boot_ci(rho, np.median, _seed_for(seed, f"trend|{c}"), cfg.n_boot)
        sc = scaling_exponent(_deltas(a, delta_mode), a["life"],
                              _seed_for(seed, f"beta|{c}"), cfg)
        red = float(np.median(a["red"])) if a["red"] else np.nan
        clo, chi = sc["ci"]
        fitted = bool(sc["fitted"] and np.isfinite(clo) and np.isfinite(chi))
        informative = bool(np.isfinite(lo) and lo >= cfg.rho_min)
        is_clock = (clo <= cfg.beta_null <= chi) if fitted else None
        # Above unity as well as covering it: both are time-driven, neither is cleared.
        superlinear = (clo > cfg.beta_null) if fitted else None
        covers_both = fitted and (clo <= 0 <= chi) and (clo <= cfg.beta_null <= chi)
        independent = bool((not np.isfinite(red)) or red < cfg.redundant)

        rows.append({
            "channel": c, "delta_mode": delta_mode,
            "n_units": len(a["rho"]), "n_valid": n_valid,
            "trend": float(np.nanmedian(rho)), "trend_ci_lo": lo,
            "beta": sc["beta"], "beta_ci_lo": clo, "beta_ci_hi": chi,
            "beta_fit_r2": sc["fit_r2"],
            "prognosability": prognosability(a["end"], a["start"]),
            "monotonicity": float(np.nanmedian(a["mono"])) if a["mono"] else np.nan,
            "redundancy": red, "informative": informative, "is_clock": is_clock,
            "superlinear": superlinear, "beta_fitted": fitted,
            "beta_covers_both": covers_both,
            # How many units the exponent was actually fitted on, which is not `n_units`
            # whenever a unit's excursion was zero or non-finite.
            "beta_n_units": int(sc["n"]),
            # fraction of each record that was finite, median over units: a channel scored
            # mostly on gaps should not be read like one scored on a full record
            "frac_finite": (float(np.nanmedian(a["frac_finite"]))
                            if a.get("frac_finite") else np.nan),
            "n_boot_effective": int(sc.get("n_boot_effective", 0)),
            "insufficient_data": False, "independent": independent,
            "indicator": bool(informative and fitted and not is_clock
                              and not superlinear and independent and c != lab),
        })
    return pd.DataFrame(rows)


def seed_sweep(acc: Mapping[str, dict], seeds: Sequence[int],
               label_channel=_UNSET, delta_mode: str = "endpoint",
               cfg: AuditConfig = AuditConfig()) -> pd.DataFrame:
    """Recompute every verdict under each seed and report which ones move.

    Both `informative` and `is_clock` compare a bootstrap interval against a fixed
    threshold, so both depend on the seed. A count that is not identical across seeds
    should be reported as a range, not an integer.
    """
    frames = []
    for s in seeds:
        t = l3_verdicts(acc, s, label_channel=label_channel,
                        delta_mode=delta_mode, cfg=cfg)
        t["seed"] = s
        frames.append(t)
    allv = pd.concat(frames, ignore_index=True)

    out = []
    for ch, g in allv.groupby("channel"):
        for test, col in (("informative", "informative"), ("clock", "is_clock"),
                          ("superlinear", "superlinear"), ("indicator", "indicator")):
            vals = [bool(v) for v in g[col].tolist() if v is not None]
            if not vals:
                continue
            k, n = sum(vals), len(vals)
            out.append({"channel": ch, "delta_mode": delta_mode,
                        "test": test, "n_seeds": n, "n_true": k,
                        "modal_verdict": k > n - k, "modal_frac": max(k, n - k) / n,
                        "unanimous": max(k, n - k) == n})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- control
def positive_control(cfg: AuditConfig = AuditConfig(), n_units: int = 60,
                     seed: int = 1) -> pd.DataFrame:
    """Score synthetic channels whose exponents are known by construction.

    A test that cannot return one of its two verdicts is indistinguishable from a test that
    always returns the other. Run this before interpreting anything: the clock must be
    flagged, the indicator must be cleared, and every interval must cover its constructed
    value.

    `super_clock` has beta = 2 by construction. Its excursion is a function of elapsed time
    alone, so it carries no damage information and must NOT be certified an indicator. A
    control built only from exponents in [0, 1] cannot detect a two-sided clearance rule
    letting it through, which is why it is here.
    """
    rng = np.random.default_rng(2026)
    lifetimes = rng.integers(100, 1600, size=n_units).astype(float)
    units, life = {}, {}
    for u, L in enumerate(lifetimes):
        n = int(L)
        t = np.arange(n, dtype=float)
        p = t / (n - 1)
        units[u] = pd.DataFrame({
            "pure_clock": t + rng.normal(0, 0.01 * max(L, 1), n),       # beta = 1
            "pure_indicator": 100.0 * p + rng.normal(0, 1.0, n),        # beta = 0
            "sqrt_scaling": np.sqrt(L) * p + rng.normal(0, 0.05 * np.sqrt(L), n),
            "super_clock": t ** 2 + rng.normal(0, 0.01 * max(L, 1) ** 2, n),   # beta = 2
        })
        life[u] = float(L)
    acc = l3_accumulate(units, life, label_channel=None, cfg=cfg)
    return l3_verdicts(acc, seed, label_channel=None, cfg=cfg)


if __name__ == "__main__":
    # channel -> (constructed beta, must be certified an indicator?)
    expected = {"pure_clock": (1.0, False), "pure_indicator": (0.0, True),
                "sqrt_scaling": (0.5, True), "super_clock": (2.0, False)}
    # The constructed exponents are exact only in the continuum. Trimming a fixed FRACTION
    # of an integer number of rows, and averaging over a window of int(len/20), leaves an
    # edge effect that grows weakly with record length: for super_clock the realised ratio
    # |delta| / L**2 drifts from 0.9408 at L=100 to 0.9456 at L=1599, tilting the true
    # exponent to about 2.0015 even with the noise set to zero. TOL absorbs that. It is a
    # property of the construction, not slack in the estimator: the same tolerance applied
    # to a wrong verdict would not rescue it, because the verdict is asserted separately.
    TOL = 0.01
    table = positive_control()
    print(table[["channel", "n_units", "beta", "beta_ci_lo", "beta_ci_hi",
                 "is_clock", "superlinear", "indicator"]].to_string(index=False))
    print()
    ok = True
    for _, r in table.iterrows():
        e, want_ind = expected[r.channel]
        covered = bool(r.beta_ci_lo - TOL <= e <= r.beta_ci_hi + TOL)
        correct = bool(r.indicator) == want_ind
        ok &= covered and correct
        print(f"  {r.channel:16s} expected beta {e:.1f}  got {r.beta:+.4f} "
              f"[{r.beta_ci_lo:+.4f}, {r.beta_ci_hi:+.4f}]  covers: {covered}  "
              f"indicator: {bool(r.indicator)} (want {want_ind})  "
              f"{'OK' if covered and correct else 'FAIL'}")
    print("\nCONTROL PASSED" if ok else "\nCONTROL FAILED, do not interpret anything")
    raise SystemExit(0 if ok else 1)
