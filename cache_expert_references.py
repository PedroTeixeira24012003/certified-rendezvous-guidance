"""
cache_expert_references.py
Cache the expert's per-trajectory results for the paired protocol (tests E2
and E3). It reuses the validated expert module for everything, same solver,
dynamics, sampler, seed and draw order, and saves to expert_refs.npz the full
worst-case trajectory for E2 and, for each of the 1000 Monte Carlo
conditions, the initial condition and the expert's figures of merit, so
every later network metric can be formed as a per-condition difference
against the expert on the identical start. Refuses to overwrite an existing
reference file.
"""

import os
import sys
import time
import numpy as np

import th_mpc_behind_1000run as EXP

OUTFILE  = 'expert_refs.npz'
N_MC     = 1000        # paired Monte Carlo conditions (matches the validation study)
MC_SEED  = 0           # SAME seed the validation study used -> identical conditions

# the per-condition figures of merit to cache for the paired deltas of E3
FOM_KEYS = ['final_pos_err', 'final_vel', 'min_dist_target', 'koz_margin',
            'koz_ok', 'peak_speed', 'vcap_viol', 'fuel', 'effort',
            'settling_time', 'converged', 'solve_fail']


def worst_case_ic():
    """The exact worst-case corner initial condition of the validation study:
    near face of the cylinder, full off-axis radius split 45 deg between R-bar
    and H-bar, residual velocity directed inward, highest eccentricity, perigee
    start. Reproduced here bit-for-bit from the expert module's constants."""
    phi = np.deg2rad(45.0)
    x0 = np.array([-90.0,
                   EXP.IC_RHO_MAX * np.cos(phi),
                   EXP.IC_RHO_MAX * np.sin(phi),
                   +EXP.IC_V_MAX,                       # inward along V-bar
                   -EXP.IC_V_MAX * np.cos(phi),         # inward on R-bar
                   -EXP.IC_V_MAX * np.sin(phi)])        # inward on H-bar
    return x0, 0.0, EXP.A_SMA_NOM, EXP.ECC_NOM          # x0, theta0, a, e


def cache_worstcase(solver):
    """Fly the expert from the worst-case corner and return its full trajectory."""
    x0, theta0, a, e = worst_case_ic()
    res = EXP.run_closed_loop(solver, x0[:6], theta0, a, e,
                              stop_on_arrival=False, verbose=False)
    return res, (x0, theta0, a, e)


def cache_paired(solver, n=N_MC, seed=MC_SEED):
    """Reproduce the validation study's 1000 conditions IN ORDER (same seed, same
    draw sequence) and record, per condition, the IC and the expert's figures of
    merit. The draw order here is byte-identical to run_randomized in the expert
    module, so trajectory i matches trajectory i in the network's E3 run."""
    rng = np.random.default_rng(seed)

    ic_x0     = np.zeros((n, 6), dtype=np.float64)   # relative state IC
    ic_theta0 = np.zeros(n, dtype=np.float64)
    ic_a      = np.zeros(n, dtype=np.float64)
    ic_e      = np.zeros(n, dtype=np.float64)
    fom = {k: np.zeros(n, dtype=np.float64) for k in FOM_KEYS}

    t0 = time.time()
    for i in range(n):
        # SAME draw sequence as run_randomized: cylinder IC, theta0, a, e
        x0 = EXP.sample_cylinder_ic(rng)
        theta0 = rng.uniform(0, 2 * np.pi)
        a = rng.uniform(*EXP.A_RANGE)
        e = rng.uniform(*EXP.E_RANGE)

        res = EXP.run_closed_loop(solver, x0[:6], theta0, a, e,
                                  stop_on_arrival=False, verbose=False)

        ic_x0[i]     = x0[:6]
        ic_theta0[i] = theta0
        ic_a[i]      = a
        ic_e[i]      = e
        for k in FOM_KEYS:
            v = res[k]
            # settling_time can be nan; converged/koz_ok are bool -> store as float
            fom[k][i] = float(v) if not isinstance(v, bool) else float(bool(v))

        if (i + 1) % 50 == 0:
            rate = (i + 1) / (time.time() - t0 + 1e-9)
            eta = (n - i - 1) / (rate + 1e-9) / 60.0
            print(f"    {i + 1}/{n}  rate {rate:.2f} traj/s  eta {eta:.0f} min")

    return ic_x0, ic_theta0, ic_a, ic_e, fom


# ============================================================================
if __name__ == "__main__":
    if os.path.exists(OUTFILE):
        print(f"ERROR: {OUTFILE} already exists.")
        print("The expert references are the fixed baseline every later comparison")
        print("is made against. Delete the file manually if you truly intend to")
        print("regenerate them (invalidating every paired result so far).")
        sys.exit(1)

    print("=" * 66)
    print("CACHING EXPERT REFERENCES FOR THE PAIRED PROTOCOL (E2, E3)")
    print("=" * 66)

    # verify the frame once, exactly as the expert does, before a long run
    EXP.verify_frame()
    print("=" * 66)

    solver = EXP.build_solver()

    # ---- (A) worst-case trajectory for E2 ----
    print("  caching worst-case trajectory (E2)...")
    wc_res, wc_ic = cache_worstcase(solver)
    X_wc = wc_res['X']            # (steps+1, 7)
    U_wc = wc_res['U']            # (steps,   3)
    print(f"    worst-case flown: {U_wc.shape[0]} steps, "
          f"converged={wc_res['converged']}, "
          f"final err {wc_res['final_pos_err']:.4f} m, "
          f"min dist {wc_res['min_dist_target']:.3f} m")

    # ---- (B) paired 1000-condition references for E3 ----
    print(f"  caching {N_MC} paired conditions (E3), seed {MC_SEED}...")
    ic_x0, ic_theta0, ic_a, ic_e, fom = cache_paired(solver)

    # sanity: the cached aggregates should match the known validation result
    conv = fom['converged'].mean()
    koz  = fom['koz_ok'].mean()
    print(f"    cached aggregate check: converged {conv*100:.1f}%  "
          f"keep-out ok {koz*100:.1f}%  "
          f"(expect 100.0% / 100.0% from the validation study)")

    # ---- save everything ----
    save_dict = dict(
        # E2 worst-case
        wc_X=X_wc.astype(np.float32), wc_U=U_wc.astype(np.float32),
        wc_x0=np.asarray(wc_ic[0], dtype=np.float64),
        wc_theta0=np.float64(wc_ic[1]),
        wc_a=np.float64(wc_ic[2]), wc_e=np.float64(wc_ic[3]),
        # E3 paired ICs
        mc_x0=ic_x0, mc_theta0=ic_theta0, mc_a=ic_a, mc_e=ic_e,
        mc_seed=MC_SEED, n_mc=N_MC,
    )
    # E3 per-condition figures of merit, one array each, prefixed fom_
    for k, arr in fom.items():
        save_dict[f'fom_{k}'] = arr

    np.savez_compressed(OUTFILE, **save_dict)

    print("=" * 66)
    print(f"  saved -> {OUTFILE}")
    print("  contents:")
    print(f"    E2 worst-case : wc_X {X_wc.shape}, wc_U {U_wc.shape}, wc_x0/theta0/a/e")
    print(f"    E3 paired     : mc_x0 {ic_x0.shape}, mc_theta0/a/e, "
          f"fom_* ({len(FOM_KEYS)} arrays of {N_MC})")
    print("  These are the fixed baseline for every paired network comparison.")
    print("  Do not regenerate.")
    print("=" * 66)