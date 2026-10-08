"""
diagnose_dgp_confounds.py — verify two things before writing the robustness narrative:
  1. Pairing premise: does heavy_tail share the exact same theta/event_times as
     baseline (and combined vs nonlinear)? This is the precondition for the
     subject-level paired delta test.
  2. Nonlinear confound mechanism: does the sin perturbation shift theta's MEAN
     (or its VARIANCE)? This decides the correct diagnostic direction.
"""
import numpy as np
from simulate_data import simulate

GAMMA = 0.8

print("=" * 70)
print("1. PAIRING PREMISE (same seed => same theta, same event_times)")
print("=" * 70)

base = simulate(seed=42, theta_mode="linear", obs_noise="gaussian", verbose=False)
heavy = simulate(seed=42, theta_mode="linear", obs_noise="heavy_tail", verbose=False)
nonlin = simulate(seed=42, theta_mode="nonlinear", obs_noise="gaussian", verbose=False)
comb = simulate(seed=42, theta_mode="nonlinear", obs_noise="heavy_tail", verbose=False)

print(f"baseline vs heavy_tail:  theta identical = {np.array_equal(base.theta, heavy.theta)}")
print(f"baseline vs heavy_tail:  event_times identical = {np.array_equal(base.event_times, heavy.event_times)}")
print(f"baseline vs heavy_tail:  event_rate = {base.event_indicators.mean():.4f} vs {heavy.event_indicators.mean():.4f}")

print(f"nonlinear vs combined:   theta identical = {np.array_equal(nonlin.theta, comb.theta)}")
print(f"nonlinear vs combined:   event_times identical = {np.array_equal(nonlin.event_times, comb.event_times)}")

print(f"baseline vs nonlinear:   theta identical = {np.array_equal(base.theta, nonlin.theta)}  (expected False)")

print()
print("=" * 70)
print("2. NONLINEAR CONFOUND MECHANISM (mean vs variance)")
print("=" * 70)

print(f"\n{'mode':<12} {'theta.mean()':>14} {'theta.std()':>14} {'E[sin(theta)]':>15}")
print("-" * 56)
for name, d in [("linear", base), ("nonlinear", nonlin)]:
    print(f"{name:<12} {d.theta.mean():>14.5f} {d.theta.std():>14.5f} "
          f"{np.mean(np.sin(d.theta)):>15.6f}")

print(f"\nNote: theta[0] ~ N(0, 0.1); sin is an ODD function => for a symmetric")
print(f"distribution E[sin(theta)] = 0 exactly, and E[theta] stays 0. The sin")
print(f"term does NOT shift the mean; it amplifies the VARIANCE (theta and")
print(f"sin(theta) are positively correlated, compounding each step).")

print()
print("=" * 70)
print("3. VARIANCE AMPLIFICATION + EXP-CONVEXITY -> EVENT RATE")
print("=" * 70)

var_ratio = (nonlin.theta.std() / base.theta.std()) ** 2
print(f"Variance ratio nonlinear/linear = {var_ratio:.3f}")
print(f"E[exp(gamma*theta)]  linear    = {np.mean(np.exp(GAMMA * base.theta)):.5f}")
print(f"E[exp(gamma*theta)]  nonlinear = {np.mean(np.exp(GAMMA * nonlin.theta)):.5f}")
print(f"Event rate  linear    = {base.event_indicators.mean():.4f}")
print(f"Event rate  nonlinear = {nonlin.event_indicators.mean():.4f}")
print()
print(f"Interpretation: hazard ∝ exp(gamma*theta). For theta ~ N(0, sigma^2),")
print(f"E[exp(gamma*theta)] = exp(gamma^2 * sigma^2 / 2). Larger sigma^2 -> larger")
print(f"mean hazard -> higher event rate. This is Jensen (exp is convex), NOT a")
print(f"mean shift. The correct diagnostic is theta.std(), not theta.mean() and")
print(f"not E[sin(theta)] (which is identically 0).")