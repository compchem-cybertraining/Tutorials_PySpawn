# -*- coding: utf-8 -*-
## Plot S1 population decay curve with bootstrap envelope and a piecewise-
## exponential fit overlay, journal-paper style.  ETHYLENE training version.

from __future__ import print_function
import matplotlib
matplotlib.use("Agg")
import os
import sys
import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

AU_TO_FS = 0.0241888425  # 1 atomic time unit in femtoseconds

def curve_fit(f, xdata, ydata, p0, bounds=None, maxfev=20000):
    x = np.asarray(xdata, dtype=float)
    y = np.asarray(ydata, dtype=float)
    p = np.array(p0, dtype=float)
    n = p.size

    if bounds is None:
        lo = np.full(n, -np.inf)
        hi = np.full(n,  np.inf)
    else:
        lo = np.array(bounds[0], dtype=float)
        hi = np.array(bounds[1], dtype=float)
        if lo.size == 1:
            lo = np.full(n, lo[0])
        if hi.size == 1:
            hi = np.full(n, hi[0])

    def clamp(pp):
        return np.minimum(np.maximum(pp, lo), hi)

    def residuals(pp):
        return f(x, *pp) - y

    p = clamp(p)
    lam = 1e-3                      # LM damping
    r = residuals(p)
    cost = float(np.dot(r, r))
    nfev = 1

    for _ in range(200):
        # Numerical Jacobian (forward differences with relative step).
        J = np.zeros((x.size, n))
        for k in range(n):
            step = 1e-6 * max(1.0, abs(p[k]))
            pk = p.copy()
            pk[k] += step
            J[:, k] = (residuals(pk) - r) / step
            nfev += 1
            if nfev > maxfev:
                break

        JTJ = J.T.dot(J)
        JTr = J.T.dot(r)

        improved = False
        for _inner in range(30):
            A = JTJ + lam * np.diag(np.diag(JTJ) + 1e-12)
            try:
                dp = np.linalg.solve(A, -JTr)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            p_new = clamp(p + dp)
            r_new = residuals(p_new)
            nfev += 1
            cost_new = float(np.dot(r_new, r_new))
            if cost_new < cost:
                # Accept step, decrease damping.
                p, r, cost = p_new, r_new, cost_new
                lam = max(lam * 0.3, 1e-12)
                improved = True
                break
            else:
                lam *= 10.0
            if nfev > maxfev:
                break

        if not improved:
            break
        if np.max(np.abs(dp)) < 1e-10 * (1.0 + np.max(np.abs(p))):
            break
        if nfev > maxfev:
            break

    # Approximate covariance: sigma^2 * (J^T J)^-1.
    try:
        dof = max(x.size - n, 1)
        sigma2 = cost / dof
        pcov = sigma2 * np.linalg.inv(JTJ)
    except Exception:
        pcov = np.full((n, n), np.nan)

    return p, pcov

plt.rc('font', family='serif')

def read_csv_columns(path):
    """Read a CSV with a header row into a dict of column_name -> list of str."""
    with open(path, "r") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        cols = dict((h, []) for h in header)
        for row in reader:
            if not row:
                continue
            for h, v in zip(header, row):
                cols[h].append(v)
    return cols


def pivot_per_sim(path, value_col, scale):
    """Read a long-format per-sim CSV (Time, Sim, value_col) and pivot to a
    wide array of shape (nsim, ntime). Returns (values, times_scaled)."""
    cols = read_csv_columns(path)
    times_raw = [float(x) for x in cols["Time"]]
    sims_raw  = [int(x)   for x in cols["Sim"]]
    vals_raw  = [float(x) for x in cols[value_col]]

    times = sorted(set(times_raw))
    sims  = sorted(set(sims_raw))
    t_index = dict((t, i) for i, t in enumerate(times))
    s_index = dict((s, i) for i, s in enumerate(sims))

    wide = np.full((len(sims), len(times)), np.nan, dtype=float)
    for tt, ss, vv in zip(times_raw, sims_raw, vals_raw):
        wide[s_index[ss], t_index[tt]] = vv

    return wide, np.array(times, dtype=float) * scale


# -----------------------------------------------------------------------------
# Model  (t is in fs throughout, tau in fs, t0 in fs)
# -----------------------------------------------------------------------------
def model_s1(t, t0, tau):
    """N_S1(t): plateau at 1 until t0, then exp(-(t-t0)/tau)."""
    t = np.asarray(t, dtype=float)
    return np.where(t < t0, 1.0, np.exp(-(t - t0) / tau))


def _initial_guess(t, y):
    """Crude initial guesses for t0, tau."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    near_one = np.where(y > 0.99)[0]
    t0_0 = float(t[near_one[-1]]) if len(near_one) else float(t[0])
    t0_0 = min(t0_0, float(t[-1]) - 1.0)

    after = y[t >= t0_0]
    if len(after) > 1:
        target = 1.0 / np.e
        below  = np.where(after <= target)[0]
        if len(below):
            tau0 = float(t[t >= t0_0][below[0]] - t0_0)
        else:
            tau0 = float(t[-1] - t0_0)
        tau0 = max(tau0, 1.0)
    else:
        tau0 = max(float(t[-1] - t0_0), 1.0)

    return [t0_0, tau0]


def fit_s1(t, y, p0=None, fix_t0=None):
    """Fit S1 model. If fix_t0 is given, t0 is held at that value and only tau is fit."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y) & (y > 0)
    if mask.sum() < 5:
        return None

    if fix_t0 is not None:
        # Single-parameter fit: only tau, with t0 pinned.
        t0 = float(fix_t0)
        # Fit only on t >= t0 to avoid the trivial 1.0 plateau region influencing tau.
        post = mask & (t >= t0)
        if post.sum() < 3:
            return None
        if p0 is None:
            p0 = [_initial_guess(t[mask], y[mask])[1]]
        elif isinstance(p0, (list, tuple)) and len(p0) > 1:
            p0 = [p0[1]]
        try:
            popt, _ = curve_fit(
                lambda tt, tau: model_s1(tt, t0, tau),
                t[post], y[post], p0=p0,
                bounds=([1e-3], [np.inf]), maxfev=20000,
            )
            return {'t0': t0, 'tau': float(popt[0]), 't0_fixed': True}
        except Exception:
            return None

    # Free t0.
    if p0 is None:
        p0 = _initial_guess(t[mask], y[mask])
    try:
        popt, _ = curve_fit(
            model_s1, t[mask], y[mask], p0=p0,
            bounds=([0.0, 1e-3], [float(t[mask][-1]), np.inf]),
            maxfev=20000,
        )
        return {'t0': float(popt[0]), 'tau': float(popt[1]), 't0_fixed': False}
    except Exception:
        return None


def evaluate_s1(t, fit):
    return model_s1(t, fit['t0'], fit['tau'])


# -----------------------------------------------------------------------------
# Bootstrap fits (CIs on parameters)
# -----------------------------------------------------------------------------
def bootstrap_fits_s1(per_sim, t, n_boot=2000, seed=12345, p0=None, fix_t0=None):
    rng = np.random.RandomState(seed)
    nsim = per_sim.shape[0]
    t0s, taus = [], []
    for _ in range(n_boot):
        idx = rng.randint(0, nsim, size=nsim)
        ymean = per_sim[idx].mean(axis=0)
        fr = fit_s1(t, ymean, p0=p0, fix_t0=fix_t0)
        if fr is not None:
            t0s.append(fr['t0']); taus.append(fr['tau'])
    return np.array(t0s), np.array(taus)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def style_axes(ax, xlim=None):
    ax.tick_params(axis='x', labelsize=18, which='major', length=6, width=1.2)
    ax.tick_params(axis='y', labelsize=18, which='major', length=6, width=1.2)
    ax.tick_params(axis='x', which='minor', length=3, width=1.0)
    ax.tick_params(axis='y', which='minor', length=3, width=1.0)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_ylim(0.0, 1.05)
    if xlim is not None:
        ax.set_xlim(*xlim)
    for axis in ['left', 'right', 'top', 'bottom']:
        ax.spines[axis].set_linewidth(1.5)


def make_plot(t, mean, lo, hi, fit_y, ylabel, out_path, color, xlim=None):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    ax.fill_between(t, lo, hi, color=color, alpha=0.35, linewidth=0)
    ax.plot(t, mean, color=color, lw=2.2)
    if fit_y is not None:
        ax.plot(t, fit_y, 'k--', lw=2.0)

    ax.set_xlabel("Time (fs)", fontsize=18)
    ax.set_ylabel(ylabel,      fontsize=18)
    style_axes(ax, xlim=xlim)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Wrote", out_path)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Plot ethylene S1 decay (au -> fs) with a piecewise-exp fit."
    )
    parser.add_argument("--summary", type=str, default="PopulationDecay_summary.csv")
    parser.add_argument("--s1-csv",  type=str,
                        default="BootStraping_time_Vs_S1_Pop_Each_Sim.csv")
    parser.add_argument("--time-unit", type=str, default="fs", choices=["fs", "au"],
                        help="Output time unit. Input CSVs are always assumed au.")

    # ---- Fit window (in plot units, default fs) ----
    parser.add_argument("--fit-tmin", type=float, default=None,
                        help="Lower bound for the S1 fit window.")
    parser.add_argument("--fit-tmax", type=float, default=None,
                        help="Upper bound for the S1 fit window.")

    # ---- x-axis limits ----
    parser.add_argument("--xlim", type=float, nargs=2, default=None,
                        help="X-axis limits for the plot.")

    # ---- Optional fixed t0 for fitting ----
    parser.add_argument("--fix-t0", type=float, default=None,
                        help="If given, pin t0 of the S1 fit to this value (in plot units).")

    parser.add_argument("--boot-fit", action="store_true",
                        help="Compute bootstrap CIs on fit parameters by refitting "
                             "resampled mean curves. Requires the per-sim CSV.")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed",   type=int, default=12345)
    parser.add_argument("--ci",     type=float, default=95.0)
    parser.add_argument("--outdir", type=str, default=".")
    parser.add_argument("--s1-color", type=str, default="tab:blue")
    args = parser.parse_args()

    scale = AU_TO_FS if args.time_unit == "fs" else 1.0
    unit  = args.time_unit

    cols = read_csv_columns(args.summary)
    t       = np.array(cols["Time"],    dtype=float) * scale
    s1_mean = np.array(cols["S1_mean"], dtype=float)
    s1_lo   = np.array(cols["S1_lo"],   dtype=float)
    s1_hi   = np.array(cols["S1_hi"],   dtype=float)

    # Fit window; fall back to data range.
    s1_tmin = args.fit_tmin if args.fit_tmin is not None else float(t.min())
    s1_tmax = args.fit_tmax if args.fit_tmax is not None else float(t.max())
    s1_fmask = (t >= s1_tmin) & (t <= s1_tmax)

    s1_fit = fit_s1(t[s1_fmask], s1_mean[s1_fmask], fix_t0=args.fix_t0)
    s1_fit_y = evaluate_s1(t, s1_fit) if s1_fit else None

    s1_boot = None
    if args.boot_fit:
        def load_wide(path, value_col):
            return pivot_per_sim(path, value_col, scale)

        s1_wide, t_w = load_wide(args.s1_csv, "S1_Pop")
        s1_fmask_w = (t_w >= s1_tmin) & (t_w <= s1_tmax)

        if s1_fit:
            s1_boot = bootstrap_fits_s1(
                s1_wide[:, s1_fmask_w], t_w[s1_fmask_w],
                n_boot=args.n_boot, seed=args.seed,
                p0=[s1_fit['t0'], s1_fit['tau']],
                fix_t0=args.fix_t0,
            )

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    data_xlim = (float(t.min()), float(t.max()))
    s1_xlim   = tuple(args.xlim) if args.xlim else data_xlim

    make_plot(
        t, s1_mean, s1_lo, s1_hi, s1_fit_y,
        ylabel=r"S$_1$ Population",
        out_path=os.path.join(args.outdir, "S1_PopulationDecay.png"),
        color=args.s1_color, xlim=s1_xlim,
    )

    fits_path = os.path.join(args.outdir, "ExponentialFits.txt")
    lo_p = (100.0 - args.ci) / 2.0
    hi_p = 100.0 - lo_p
    with open(fits_path, 'w') as f:
        f.write("# Piecewise-exponential fit\n")
        f.write("# Time unit: {}\n".format(unit))
        f.write("# S1 fit window: [{}, {}] {}\n\n".format(s1_tmin, s1_tmax, unit))

        f.write("=== S1 total population ===\n")
        f.write("# Model: N(t) = 1 for t<t0, exp(-(t-t0)/tau) for t>=t0\n")
        if s1_fit is None:
            f.write("  fit failed\n\n")
        else:
            t0_tag = "  (FIXED)" if s1_fit.get('t0_fixed') else ""
            f.write("  t0  = {:.4f}  ({}){}\n".format(s1_fit['t0'], unit, t0_tag))
            f.write("  tau = {:.4f}  ({})\n".format(s1_fit['tau'], unit))
            if s1_boot is not None and len(s1_boot[1]) > 10:
                t0s, taus = s1_boot
                f.write("  bootstrap ({:.0f}% CI, {} successful fits):\n".format(
                    args.ci, len(taus)))
                if not s1_fit.get('t0_fixed'):
                    f.write("    t0  CI: [{:.4f}, {:.4f}]\n".format(
                        np.percentile(t0s, lo_p), np.percentile(t0s, hi_p)))
                f.write("    tau CI: [{:.4f}, {:.4f}]\n".format(
                    np.percentile(taus, lo_p), np.percentile(taus, hi_p)))
            f.write("\n")

    print("Wrote", fits_path)


if __name__ == "__main__":
    main()
