# rul-deposit-audit

A three-leg reusability audit for public run-to-failure prognostics deposits.

It scores the **dataset**, not a model. Nothing is trained, nothing needs a GPU, and the
whole thing runs on `numpy`, `pandas` and `scipy`.

| Leg | Question | Returns |
|---|---|---|
| 1 | Is the released target recoverable from the record length alone? | R², exact-identity rate, zero-parameter error, ICC(2,1) |
| 2 | Was the failure event established independently of the channels? | grade A / B / C, documentary |
| 3 | Does an informative, independent, non-clock channel exist? | per-channel profile and verdict |

## The part that is new

Trendability rewards any quantity that rises steadily through a record, and elapsed time
rises steadily through every record. A clock passes it. So does monotonicity.

Leg 3 adds a scaling test. Fit `|Δ| = κ · lifetime^β` across units:

* `β → 1` the channel accumulates with elapsed time. It is a **clock**. A unit that survived
  twice as long moved twice as far.
* `β → 0` the channel reaches a comparable end state however long the unit lived. It is a
  **damage indicator**.

The null is `β = 1`, fixed by the two hypotheses rather than by any observed value.

**Clearance is one-sided.** A channel is cleared only when the whole bootstrap interval on β
lies **below** unity. Not merely when it excludes unity, which is a different and wrong rule:
an interval lying entirely *above* unity also excludes it, and a channel whose excursion
grows faster than the lifetime is driven by elapsed time more strongly than a clock is.
Doubling a unit's survival time more than doubles how far the channel moved, so its terminal
value is set by the timing rather than in spite of it. Both cases block clearance and are
reported separately, as `is_clock` when the interval covers unity and `superlinear` when it
sits above. The control includes a `β = 2` channel for exactly this reason: a control built
only from exponents in `[0, 1]` cannot tell the two rules apart.

## Run the control first

```bash
pip install -r requirements.txt   # scipy >= 1.10 is required
python rul_audit.py
```

`scipy >= 1.10` is not optional: the code reads `.statistic` off a `spearmanr` result, which
older releases do not provide.

This scores synthetic channels whose exponents are known by construction. A test that cannot
return one of its two verdicts is indistinguishable from a test that always returns the
other, so if the control does not pass, nothing downstream means anything.

## Auditing your own deposit

```python
import pandas as pd
from rul_audit import AuditConfig, leg1, l3_accumulate, l3_verdicts, seed_sweep

cfg = AuditConfig()

# Leg 1 needs two columns: one released target and one record length per unit.
print(leg1(targets=[...], lengths=[...], cfg=cfg))

# Leg 3 needs the signals. `units` maps a unit id to a DataFrame of channels;
# `lifetimes` maps the same ids to that unit's lifetime.
# A `label_channel` that matches no column raises rather than silently disabling the
# independence check, and the accumulator remembers it so the verdict step cannot be
# handed a different one.
acc = l3_accumulate(units, lifetimes, label_channel="acc_horiz_rms", cfg=cfg)
print(l3_verdicts(acc, seed=1, cfg=cfg))
print(l3_verdicts(acc, seed=1, delta_mode="raw", cfg=cfg))   # sensitivity

# Both Leg 3 criteria compare a bootstrap interval to a threshold, so both depend on
# the seed. Sweep it, and report any count that moves as a range rather than an integer.
print(seed_sweep(acc, seeds=range(1, 21), cfg=cfg))
```

## Two design choices worth knowing

**Accumulate and score are separate.** `l3_accumulate` is expensive and deterministic;
`l3_verdicts` is cheap and seed-dependent. Checkpoint the accumulator and a seed sweep, a
changed threshold, or a new criterion costs no access to the raw signals.

**Rows are read positionally.** Progress through a record is row index over row count, so a
frame that is not already in chronological order yields a plausible-looking but meaningless
trend. No index is consulted for ordering. A non-monotonic `DatetimeIndex` is refused, which
catches the common case and not the general one, so sort before calling.

**Gappy channels are scored and flagged, not dropped.** `frac_finite` in the verdict table is
the median fraction of each record that was finite for that channel. A channel scored on five
percent of its samples gets a verdict like any other, and that column is how you notice.

**An absent fit is not a verdict.** With fewer than five usable units, or with no spread in
lifetime, the exponent is not fitted and `is_clock` is `None` rather than `True`. Treating a
failed fit as a clock fails closed and prints a confident answer from no evidence.

**The excursion is measured three ways.** `l3_accumulate` writes all of them, and
`delta_mode` selects which one `l3_verdicts` scores:

| mode | `|Δ|` is | note |
|---|---|---|
| `endpoint` | last smoothed value minus the first | **the default and the specified definition** |
| `smoothed` | mean of the first and last `k` smoothed values | cleaner start, but the end spans `2k-1` raw samples and `k` grows with the record, which biases β down in long units |
| `raw` | mean of the first and last `k` raw samples | equal windows at both ends |

They disagree. On the deposits in the paper the median absolute change in β between
`endpoint` and `smoothed` is 0.15. Pick one before you look at the answer, report the others
as sensitivity, and do not let the choice be made by which one gives a nicer result.

**The exponent comes with the quality of its own fit.** `beta_fit_r2` is the R² of the
log-log regression. A percentile bootstrap assumes no functional form, so a low value does
not invalidate the interval, but it does mean the excursion carries little systematic
dependence on lifetime and the point estimate should not be read as if it did.

## Thresholds

All fixed before scoring, and all in `AuditConfig`:

| Setting | Default | Role |
|---|---|---|
| `l1_fail_r2` | 0.90 | at or above, the target is recoverable from the record |
| `rho_min` | 0.30 | trendability cut; the interval's lower bound must clear it |
| `redundant` | 0.80 | \|ρ\| against the target-defining channel |
| `beta_null` | 1.0 | the clock hypothesis |
| `min_units_for_beta` | 5 | below this no interval is produced |
| `burn_in` | 0.10 | leading fraction of each record discarded |
| `n_boot` | 10000 | bootstrap resamples, percentile intervals, resampling units |

## What the exponent identifies, and what it does not

`analysis/` holds two scripts that bound what a clearance is worth. Both are CPU-only and
their simulation parts need no dataset at all.

`threshold_identifiability.py` settles what the exponent measures. Write a unit's lifetime
as the point where accumulated damage reaches that unit's threshold, `L = theta / r`, and
let the channel's excursion be the damage accumulated, `Delta = theta`. The log-log slope
is then

    beta  =  var(log theta) / (var(log theta) + var(log r))  ==  omega

the share of log-lifetime variance coming from threshold spread rather than rate spread.
So `omega = 0` is a damage indicator and `omega = 1` is a channel that genuinely tracks
damage on a rig with a random critical flaw size, which this test cannot tell from a clock.
The script confirms the algebra, then measures what a real panel resolves: at 17 units a
channel with `omega = 0` is cleared every time, while one with `omega = 1` is called a
clock 92.8% of the time. That is a limit of identification, not of sample size.

It also measures the second gap. A saturating transient `1 - exp(-t/tau)` depends on
elapsed time and on nothing about the unit, yet it is cleared in 100% of panels when `tau`
is a quarter of the median lifetime. **A clearance rules out a linear clock and nothing
more.** A cleared channel still has to earn a physical reading from evidence outside this
test.

`interval_method.py` asks whether the verdict depends on how the interval is built. It
scores a percentile, a BCa and an ordinary OLS-`t` interval on the same bootstrap draws,
under three residual regimes, two of which deliberately violate the `t` interval's own
assumptions. Bias correction does **not** repair the small-sample under-coverage: worst
cells are 0.887 percentile, 0.883 BCa, 0.915 OLS-`t` against a nominal 0.95. It also
rescores the panel under a Bonferroni-equivalent widening to `1 - 0.05/m`, the honest
answer to a per-channel scan that produces no p-value to adjust.

Both scripts read per-unit arrays for the parts that rescore a real panel. Those arrays
are derived from four public deposits and are not redistributed here, so those sections
need you to run your own extraction first; the simulation sections run standalone.

## Citation

If you use this, please cite the accompanying paper. Citation details will be added here on
acceptance.

## Licence

MIT.
