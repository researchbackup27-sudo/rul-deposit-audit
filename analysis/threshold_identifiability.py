"""
nb15: bound the threshold-heterogeneity confound instead of only disclosing it.

The Limitations section concedes that where lifetimes differ because the failure THRESHOLD
varies between units rather than because units accumulate damage at different RATES, a
genuine damage channel returns beta = 1 exactly and is recorded here as a clock.

That concession is qualitative. This quantifies it. Under a generator in which both the
per-unit accumulation rate r_u and the per-unit failure threshold theta_u vary,

    L_u = theta_u / r_u      (unit fails when accumulated damage reaches its threshold)
    Delta_u = theta_u        (the channel's excursion IS the damage accumulated)

so with log r and log theta independent,

    beta = Cov(log theta, log theta - log r) / Var(log theta - log r)
         = sigma_theta^2 / (sigma_theta^2 + sigma_r^2)  ==  omega

beta therefore estimates omega, the FRACTION OF LOG-LIFETIME VARIANCE contributed by
threshold heterogeneity. omega = 1 is the pure random-critical-flaw-size picture (a real
damage channel that is nonetheless redundant with the record length), omega = 0 is the
textbook damage indicator (all units share a threshold, lifetimes differ by rate).

Three things are measured:
  (1) that beta really does estimate omega, at large n, against the algebra;
  (2) at n = 17 (the FEMTO panel), P(clock verdict) as a function of omega -- the power
      curve the review asked for;
  (3) what omega the observed FEMTO interval is actually consistent with.

CPU only. No dataset. Nothing is read from the deposits.
"""
import json, os, sys, numpy as np
from pathlib import Path
sys.path.insert(0, ".")
import rul_audit as R

OUT = Path("threshold_confound"); OUT.mkdir(exist_ok=True)
cfg = R.AuditConfig()
rng = np.random.default_rng(20260921)
SIGMA_TOT = 0.717         # sd(log lifetime) MEASURED on the FEMTO panel, asserted below
# Lognormal noise on the excursion. This is NOT a free parameter: it sets how well the
# slope is determined, so leaving it at a convenient small value would manufacture power
# the real panel does not have. It is calibrated to the median residual scale of the
# FEMTO log-log fits, the same panel SIGMA_TOT is calibrated to, and asserted below.
NOISE     = 1.132


def generate(n, omega, sigma_tot=SIGMA_TOT, noise=NOISE, rng=rng):
    """One synthetic deposit: n units, a damage channel, threshold fraction omega."""
    s_theta = sigma_tot * np.sqrt(omega)
    s_r     = sigma_tot * np.sqrt(1.0 - omega)
    log_theta = rng.normal(0.0, s_theta, n) if s_theta > 0 else np.zeros(n)
    log_r     = rng.normal(0.0, s_r,     n) if s_r     > 0 else np.zeros(n)
    L     = np.exp(6.0 + log_theta - log_r)                     # theta / r
    delta = np.exp(log_theta + rng.normal(0.0, noise, n))       # the channel excursion
    return delta, L


def head(t): print("\n" + "=" * 78); print(t); print("=" * 78)

# ------------------------------------------------------------------ (1) beta estimates omega
head("(1) does beta recover omega? large-n check against the algebra")
# The estimator is unbiased for omega, not exact on any one draw, so a single fit per
# omega tests a realisation rather than the estimator. Average over replicates and judge
# the mean against its own Monte Carlo standard error.
REPS = 40
print(f"  {REPS} replicates of n=4000 per omega")
print(f"{'omega':>7s}{'mean beta':>12s}{'MC se':>9s}{'|bias|':>9s}{'bias/se':>9s}")
rec = {}
worst_z = 0.0
for omega in [0.0, 0.25, 0.5, 0.75, 1.0]:
    bs = []
    for _ in range(REPS):
        d, L = generate(4000, omega)
        bs.append(R.scaling_exponent(d, L, seed=1, cfg=R.AuditConfig(n_boot=2))["beta"])
    bs = np.asarray(bs)
    mean, se = float(bs.mean()), float(bs.std(ddof=1) / np.sqrt(REPS))
    bias = abs(mean - omega)
    z = bias / se if se > 0 else 0.0
    worst_z = max(worst_z, z)
    rec[str(omega)] = {"mean_beta": mean, "mc_se": se, "bias": bias, "z": z}
    print(f"{omega:7.2f}{mean:12.4f}{se:9.4f}{bias:9.4f}{z:9.2f}")
algebra_ok = worst_z < 4.0
print(f"\n  worst |bias| / MC se = {worst_z:.2f}  ->  "
      f"{'beta estimates omega' if algebra_ok else 'ALGEBRA NOT CONFIRMED'}")
worst_bias = max(v["bias"] for v in rec.values())
print(f"  largest absolute bias across the grid = {worst_bias:.4f}")

# ------------------------------------------------------------------ (2) the power curve
head("(2) at n = 17, how often is a damage channel of given omega called a clock?")
N_PANEL, TRIALS = 17, 2000
GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
print(f"  {TRIALS} trials per cell, n = {N_PANEL}, sigma_logL = {SIGMA_TOT}, "
      f"excursion noise = {NOISE}")
print(f"\n{'omega':>7s}{'P(cleared)':>13s}{'P(clock)':>11s}{'P(superlin)':>13s}"
      f"{'median beta':>13s}")
curve = {}
for omega in GRID:
    cleared = clock = superlin = 0; tot = 0; betas = []
    for _ in range(TRIALS):
        d, L = generate(N_PANEL, omega)
        sc = R.scaling_exponent(d, L, seed=int(rng.integers(1 << 62)), cfg=cfg)
        if not sc["fitted"]:
            continue
        tot += 1; betas.append(sc["beta"])
        lo, hi = sc["ci"]
        if hi < 1.0:   cleared  += 1
        elif lo > 1.0: superlin += 1
        else:          clock    += 1
    if tot == 0:
        print(f"{omega:7.2f}  no trial produced a fit; widen the panel"); continue
    curve[str(omega)] = {"n": tot, "cleared": cleared / tot, "clock": clock / tot,
                         "superlinear": superlin / tot, "median_beta": float(np.median(betas))}
    print(f"{omega:7.2f}{cleared/tot:13.3f}{clock/tot:11.3f}{superlin/tot:13.3f}"
          f"{np.median(betas):13.3f}")

power_at_0 = curve["0.0"]["cleared"]
print(f"\n  POWER: a textbook damage channel (omega = 0) is cleared in "
      f"{power_at_0:.1%} of panels of 17 units.")
print(f"  A pure random-flaw-size channel (omega = 1) is called a clock in "
      f"{curve['1.0']['clock']:.1%} and cleared in {curve['1.0']['cleared']:.1%}.")

# ------------------------------------------------------------------ (4) the nonlinear clock
head("(4) how often does a NONLINEAR function of elapsed time clear the test?")
print("  The null in Equation eq:clock is linear, so a channel that depends only on elapsed")
print("  time but nonlinearly can clear it while carrying no unit-specific information.")
print("  A saturating transient v(t) = 1 - exp(-t / tau) is the realistic case: its")
print("  excursion is a deterministic function of the record length and nothing else.")
print(f"\n  n = {N_PANEL}, sigma_logL = {SIGMA_TOT}, excursion noise = {NOISE}")
print(f"\n{'tau / median life':>19s}{'P(cleared)':>13s}{'P(clock)':>11s}"
      f"{'P(superlin)':>13s}{'median beta':>13s}")
MEDLIFE = float(np.exp(6.0))
nonlin = {}
for ratio in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]:
    tau = ratio * MEDLIFE
    cleared = clock = superlin = 0; tot = 0; betas = []
    for _ in range(TRIALS):
        L = np.exp(6.0 + rng.normal(0.0, SIGMA_TOT, N_PANEL))
        # excursion of a saturating transient over a record of length L, nothing unit-specific
        d = (1.0 - np.exp(-L / tau)) * np.exp(rng.normal(0.0, NOISE, N_PANEL))
        sc2 = R.scaling_exponent(d, L, seed=int(rng.integers(1 << 62)), cfg=cfg)
        if not sc2["fitted"]:
            continue
        tot += 1; betas.append(sc2["beta"])
        lo2, hi2 = sc2["ci"]
        if hi2 < 1.0: cleared += 1
        elif lo2 > 1.0: superlin += 1
        else: clock += 1
    if tot == 0:
        print(f"{ratio:19.2f}  no trial produced a fit"); continue
    nonlin[str(ratio)] = {"n": tot, "cleared": cleared / tot, "clock": clock / tot,
                          "superlinear": superlin / tot,
                          "median_beta": float(np.median(betas))}
    print(f"{ratio:19.2f}{cleared/tot:13.3f}{clock/tot:11.3f}"
          f"{superlin/tot:13.3f}{np.median(betas):13.3f}")

worst_clear = max(v["cleared"] for v in nonlin.values())
print(f"\n  A saturating clock is CLEARED in up to {worst_clear:.1%} of panels, and the")
print("  shorter its time constant relative to the record, the more often. Once the")
print("  transient has settled the excursion stops growing with lifetime, so the exponent")
print("  falls to zero and the channel is certified on no unit-specific information.")
print("  This is a FALSE CLEARANCE the test cannot detect, and it bounds what a")
print("  clearance is worth: clearance rules out a LINEAR clock, nothing more.")
nonlin_ok = worst_clear > 0.5   # the failure mode is real and must be reported as such
print(f"\n  confound is material: {nonlin_ok}")

# ------------------------------------------------------------------ (3) invert onto FEMTO
head("(3) what omega is the observed FEMTO kurtosis interval consistent with?")
# Per-unit arrays are derived from a public deposit and are not redistributed
# here. Point RUL_ACCUMULATORS at your own extraction output to run this part;
# everything above it is simulation and needs no data.
ACC = Path(os.environ.get("RUL_ACCUMULATORS", "accumulators")) / "femto.json"
if not ACC.exists():
    print(f"  SKIPPED: no per-unit arrays at {ACC}.")
    print("  Sections (1), (2) and (4) are simulation only and have already run.")
    json.dump({"sigma_tot": SIGMA_TOT, "noise": NOISE, "trials": TRIALS,
               "n_panel": N_PANEL, "algebra_check": rec,
               "algebra_ok": bool(algebra_ok), "power_curve": curve,
               "nonlinear_clock": nonlin, "femto_acc_horiz_kurt": None,
               "calibrated": None, "noise_calibrated": None},
              open(OUT / "threshold_confound.json", "w"), indent=2)
    sys.exit(0 if (algebra_ok and nonlin_ok) else 1)
femto = json.load(open(ACC))["acc"]["acc_horiz_kurt"]
sc = R.scaling_exponent(femto["delta"], femto["life"],
                        seed=R._seed_for(1, "beta|acc_horiz_kurt"), cfg=cfg)
lo, hi = sc["ci"]
print(f"  acc_horiz_kurt  n = {sc['n']}  beta = {sc['beta']:.4f}  "
       f"CI [{lo:.4f}, {hi:.4f}]  log-log R2 = {sc['fit_r2']:.4f}")
om_lo, om_hi = max(0.0, lo), min(1.0, hi)
print(f"\n  omega is a variance fraction, so it lives in [0, 1]. Intersecting the interval")
print(f"  with that range leaves omega in [{om_lo:.3f}, {om_hi:.3f}].")
print(f"  Reading: between {om_lo:.0%} and {om_hi:.0%} of the log-lifetime variance on this")
print(f"  deposit is threshold heterogeneity rather than rate heterogeneity. The data do")
print(f"  not separate a clock from a damage channel whose units differ mainly in threshold.")

# what the observed log-lifetime spread actually is, so SIGMA_TOT is not invented
Lf = np.asarray(femto["life"], float)
resid_sds = []
for _ch, _a in json.load(open(ACC))["acc"].items():
    _d = np.abs(np.asarray(_a["delta"], float)); _L = np.asarray(_a["life"], float)
    _ok = np.isfinite(_d) & np.isfinite(_L) & (_d > 0) & (_L > 0)
    _d, _L = _d[_ok], _L[_ok]
    if len(_d) < 5:
        continue
    _x, _y = np.log(_L), np.log(_d)
    _b, _c = np.polyfit(_x, _y, 1)
    resid_sds.append(float(np.std(_y - (_b * _x + _c), ddof=2)))
noise_obs = float(np.median(resid_sds))
print(f"\n  median log-log residual sd over FEMTO channels = {noise_obs:.3f}; "
      f"simulation used {NOISE} -> "
      f"{'calibrated' if abs(noise_obs - NOISE) < 0.02 else 'MISCALIBRATED'}")
noise_ok = abs(noise_obs - NOISE) < 0.02
sd_obs = float(np.std(np.log(Lf)))
print(f"\n  sd(log lifetime) on FEMTO = {sd_obs:.3f}; simulation used {SIGMA_TOT} "
      f"-> {'calibrated' if abs(sd_obs - SIGMA_TOT) < 0.02 else 'MISCALIBRATED'}")
calib_ok = abs(sd_obs - SIGMA_TOT) < 0.02

art = {"sigma_tot": SIGMA_TOT, "noise": NOISE, "trials": TRIALS, "n_panel": N_PANEL,
       "algebra_check": rec, "algebra_ok": bool(algebra_ok), "power_curve": curve, "nonlinear_clock": nonlin,
       "femto_acc_horiz_kurt": {"n": sc["n"], "beta": sc["beta"], "ci": list(sc["ci"]),
                                "fit_r2": sc["fit_r2"], "omega_lo": om_lo, "omega_hi": om_hi},
       "femto_sd_log_life": sd_obs, "calibrated": bool(calib_ok),
       "noise_observed": noise_obs, "noise_calibrated": bool(noise_ok)}
json.dump(art, open(OUT / "threshold_confound.json", "w"), indent=2)
print(f"\nartifact -> {OUT/'threshold_confound.json'}")
sys.exit(0 if (algebra_ok and calib_ok and noise_ok and nonlin_ok) else 1)
