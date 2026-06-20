# -*- coding: utf-8 -*-
## By A. Mehmood (data-generation step) -- ETHYLENE training version (SSAIMS)

from __future__ import print_function
import sys
import h5py
import os
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count

### Default simulation parameters (ethylene training data) ###
nosims   = 50       # adjust to your number of initial conditions
maxtime  = 3500     # atomic time units
timestep = 10       # atomic time units


# -----------------------------------------------------------------------------
# Per-sim worker
# -----------------------------------------------------------------------------
def _parse_list_field(x):
    """labels_this_step / istates_this_step entries are stored as a single
    comma-joined bytes/str per step (e.g. b'00,00b0b0,00b0b0b0'). Split into
    a clean list of strings, dropping empties."""
    if isinstance(x, (bytes, np.bytes_)):
        x = x.decode('ascii')
    return [s for s in str(x).split(",") if s != ""]


def s1_pop_this_step(nt, c_row, S_row, istates):
    """Total S1 population at one saved quantum step.

    SSAIMS: the live TBFs at this step are listed in labels_this_step /
    istates_this_step, and their order IS the row/column order of the
    nt x nt overlap matrix S and the amplitude vector c. We sum the
    Mulliken-like population over every live TBF whose electronic state
    is S1 (istate == 1).
    """
    if nt == 0:
        return 0.0
    nt2 = nt * nt
    c_t = c_row[0:nt]
    S_t = S_row[0:nt2].reshape((nt, nt))

    tot = 0.0
    for a in range(nt):
        if a >= len(istates) or istates[a] != 1:
            continue  # not an S1 TBF
        pop_a = 0.0
        for b in range(nt):
            pop_a += np.real(0.5 * (
                np.conjugate(c_t[a]) * S_t[a, b] * c_t[b] +
                np.conjugate(c_t[b]) * S_t[b, a] * c_t[a]
            ))
        tot += pop_a
    return float(tot)


def process_one_sim(args):
    """
    Process one SSAIMS simulation across all timesteps.

    Returns a per-time array giving this sim's total S1 population.
    This is the NO-carry-forward variant: grid points that have no matching
    saved quantum step are left at 0 for this sim (nothing is reused).

    SSAIMS note: TBFs are spawned and killed independently, so the number of
    live TBFs (num_traj_qm) changes step to step. The per-step overlap matrix
    S and amplitude vector c are ordered to match labels_this_step /
    istates_this_step at that step -- NOT the global sim.attrs['labels'] order.
    We therefore read state populations directly from each saved step.
    """
    (cursim, simroot, maxtime, timestep) = args

    times  = list(range(0, maxtime, timestep))
    ntimes = len(times)
    tol    = 0.5 * timestep

    s1_total_per_t = np.zeros(ntimes)

    simdir = os.path.join(simroot, str(cursim))
    h5path = os.path.join(simdir, "sim.hdf5")

    if not os.path.isfile(h5path):
        raise IOError("Missing HDF5: " + h5path)

    print("Processing sim {} : {}".format(cursim, simdir))
    sys.stdout.flush()

    with h5py.File(h5path, "r") as h5file:
        qtimes = h5file["sim/quantum_time"][()][:, 0]
        ntq    = h5file["sim/num_traj_qm"][()].flatten()
        S      = h5file["sim/S"][()]
        c      = h5file["sim/qm_amplitudes"][()]
        istep  = h5file["sim/istates_this_step"][()]

        nsteps = len(qtimes)

        # Precompute the S1 total at every saved quantum step.
        step_time = np.empty(nsteps)
        step_s1   = np.empty(nsteps)
        for k in range(nsteps):
            nt = int(ntq[k])
            istates = [int(v) for v in _parse_list_field(istep[k])]
            step_time[k] = qtimes[k]
            step_s1[k]   = s1_pop_this_step(nt, c[k], S[k], istates)

        # Map each output-grid time onto the nearest saved step within tol.
        # No carry-forward: grid points with no matching saved step stay 0.0.
        for it, t in enumerate(times):
            tf = float(t)
            if nsteps > 0:
                k = int(np.argmin(np.abs(step_time - tf)))
                if abs(step_time[k] - tf) <= tol:
                    s1_total_per_t[it] = step_s1[k]
            # else / no match: leave as 0.0

    return {
        'cursim':   cursim,
        's1_total': s1_total_per_t,
    }


# -----------------------------------------------------------------------------
# Bootstrap
# -----------------------------------------------------------------------------
def bootstrap_curve(per_sim_curves, n_boot=2000, ci=95.0, seed=12345):
    """
    per_sim_curves: array (nsim, ntimes)
    Returns (mean, lo, hi) per time. Mean is the simple mean over sims; lo/hi
    are percentile CI from resampling sims with replacement.
    """
    rng = np.random.RandomState(seed)
    nsim, ntimes = per_sim_curves.shape
    boot_means = np.empty((n_boot, ntimes))
    for b in range(n_boot):
        idx = rng.randint(0, nsim, size=nsim)
        boot_means[b] = per_sim_curves[idx].mean(axis=0)
    lo_p = (100.0 - ci) / 2.0
    hi_p = 100.0 - lo_p
    mean = per_sim_curves.mean(axis=0)
    lo   = np.percentile(boot_means, lo_p, axis=0)
    hi   = np.percentile(boot_means, hi_p, axis=0)
    return mean, lo, hi


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate per-sim and bootstrap CSVs for ethylene S1 decay."
    )
    parser.add_argument("-n", "--nprocs", type=int, default=max(1, cpu_count() - 1))
    parser.add_argument("--nosims",   type=int, default=nosims)
    parser.add_argument("--maxtime",  type=int, default=maxtime)
    parser.add_argument("--timestep", type=int, default=timestep)
    parser.add_argument("--simroot",  type=str, default=None,
                        help="Path containing numbered sim dirs. Default: parent of cwd.")
    parser.add_argument("--n-boot", type=int,   default=2000)
    parser.add_argument("--ci",     type=float, default=95.0)
    parser.add_argument("--seed",   type=int,   default=12345)
    args = parser.parse_args()

    cwd     = os.getcwd()
    simroot = os.path.abspath(args.simroot) if args.simroot else os.path.dirname(cwd)
    times_arr = np.array(list(range(0, args.maxtime, args.timestep)), dtype=float)
    ntimes  = len(times_arr)

    print("Sim root  (input):  {}".format(simroot))
    print("Output dir:         {}".format(cwd))
    print("nosims={}, maxtime={}, timestep={}".format(
        args.nosims, args.maxtime, args.timestep))
#    print("Carry-forward: OFF (grid points contribute only when a saved step matches)")
    print("")

    work = [
        (sim, simroot, args.maxtime, args.timestep)
        for sim in range(1, args.nosims + 1)
    ]

    print("Launching {} workers over {} sims, {} timesteps each.".format(
        args.nprocs, args.nosims, ntimes))
    sys.stdout.flush()

    pool = Pool(processes=args.nprocs)
    try:
        results = []
        for i, r in enumerate(pool.imap_unordered(process_one_sim, work), start=1):
            results.append(r)
            print("  -> finished sim {} ({}/{})".format(r['cursim'], i, args.nosims))
            sys.stdout.flush()
    finally:
        pool.close()
        pool.join()

    results.sort(key=lambda r: r['cursim'])

    s1_curves = np.vstack([r['s1_total'] for r in results])
    sim_ids   = [r['cursim'] for r in results]

    # Per-sim long-format CSV.
    csv1_path = "BootStraping_time_Vs_S1_Pop_Each_Sim.csv"
    with open(csv1_path, 'w') as f:
        f.write("Time,Sim,S1_Pop\n")
        for it, t in enumerate(times_arr):
            for ks, sid in enumerate(sim_ids):
                f.write("{},{},{}\n".format(int(t), sid, s1_curves[ks, it]))
    print("Wrote", csv1_path)

    # Bootstrap summary.
    print("Bootstrapping ({} resamples, {}% CI)...".format(args.n_boot, args.ci))
    s1_mean, s1_lo, s1_hi = bootstrap_curve(
        s1_curves, n_boot=args.n_boot, ci=args.ci, seed=args.seed)

    summary_path = "PopulationDecay_summary.csv"
    with open(summary_path, 'w') as f:
        f.write("Time,S1_mean,S1_lo,S1_hi\n")
        for it, t in enumerate(times_arr):
            f.write("{},{:.6f},{:.6f},{:.6f}\n".format(
                int(t), s1_mean[it], s1_lo[it], s1_hi[it]))
    print("Wrote", summary_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
