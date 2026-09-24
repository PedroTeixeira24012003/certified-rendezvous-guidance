"""
expert_path_deviation.py
Per-step deviation of each network's trajectory from the expert's on the same
condition and the same disturbance tape, at the requested rungs, over whole
flights. Pairs of different length are compared up to the shorter one and the
mismatch count is reported. The expert is flown once per rung and its paths
cached to exppath_<rung>.npz, then every network is diffed against the cache.
Prints pooled step deviations and per-flight peaks, writes
expert_path_deviation_summary.csv.
"""

import os
import csv
import argparse
import numpy as np

import th_mpc_behind_1000run as EXP
import deploy_closedloop as DCL
import bc_data
import robustness_ladder as RL

MANIFEST = [
    ("ta=0.01",  "checkpoints/20260813-141246-4t-cc7328.pt"),
    ("ta=0.03",  "checkpoints/20260813-152458-4t-6fc9e3.pt"),
    ("ta=0.1",   "checkpoints/20260813-154058-4t-68990a.pt"),
    ("ta=0.3",   "checkpoints/20260813-161306-4t-cea077.pt"),
    ("ta=1.0",   "checkpoints/20260813-163542-4t-d2afae.pt"),
    ("ta=5.0",   "checkpoints/20260813-171313-4t-b1fc55.pt"),
    ("ta=8.0",   "checkpoints/20260813-183326-4t-828492.pt"),
]


def fly_expert(solver, x0, theta0, a, e, tape):
    state = np.concatenate([x0, [theta0]])
    X = [state.copy()]
    env_fn = tape.env_fn(a, e)
    p_val = np.array([a, e])
    for k in range(EXP.SIM_STEPS):
        R = np.linalg.norm(state[:3])
        meas = state.copy()
        meas[:6] = state[:6] + tape.nav_noise(R)
        for j in range(EXP.N_P + 1):
            solver.set(j, "p", p_val)
        solver.set(0, "lbx", meas)
        solver.set(0, "ubx", meas)
        solver.solve()
        u = solver.get(0, "u")
        u_app = tape.thrust_apply(u)
        state = RL.rk4_step_disturbed(state, u_app, EXP.DT, a, e, env_fn)
        X.append(state.copy())
        if not np.all(np.isfinite(state)):
            break
        if np.linalg.norm(state[:3] - EXP.HOLD) > DCL.DIVERGE_DIST:
            break
    return np.array(X)[:, :3]      # positions only


def fly_network(model, x0, theta0, a, e, x_mean, x_std, tape):
    state = np.concatenate([x0, [theta0]])
    X = [state.copy()]
    env_fn = tape.env_fn(a, e)
    for k in range(EXP.SIM_STEPS):
        R = np.linalg.norm(state[:3])
        meas = state.copy()
        meas[:6] = state[:6] + tape.nav_noise(R)
        u = DCL.policy_command(model, meas, a, e, x_mean, x_std)
        u_app = tape.thrust_apply(u)
        state = RL.rk4_step_disturbed(state, u_app, EXP.DT, a, e, env_fn)
        X.append(state.copy())
        if not np.all(np.isfinite(state)):
            break
        if np.linalg.norm(state[:3] - EXP.HOLD) > DCL.DIVERGE_DIST:
            break
    return np.array(X)[:, :3]


def build_expert_cache(rung, cfg, refs, solver):
    """Fly the expert on all conditions at this rung, store variable-length
    position paths as an object array. Cached to disk for reuse."""
    cache = f"exppath_{rung}.npz"
    if os.path.exists(cache):
        print(f"  [{rung}] expert cache found: {cache}")
        d = np.load(cache, allow_pickle=True)
        return list(d["paths"])
    n = int(refs["n_mc"])
    mc_x0 = refs["mc_x0"]; mc_th = refs["mc_theta0"]
    mc_a = refs["mc_a"]; mc_e = refs["mc_e"]
    print(f"  [{rung}] flying expert to build path cache "
          f"(~45 min)...")
    import time
    t0 = time.time()
    paths = []
    for i in range(n):
        tape = RL.DisturbanceTape(rung, i, cfg)
        paths.append(fly_expert(solver, mc_x0[i], float(mc_th[i]),
                                float(mc_a[i]), float(mc_e[i]), tape))
        if (i + 1) % 100 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"    expert {i+1}/{n}  eta {(n-i-1)/rate/60:.1f} min")
    np.savez_compressed(cache, paths=np.array(paths, dtype=object))
    print(f"  [{rung}] expert cache saved: {cache}")
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", nargs="+", default=["L2", "L3"])
    ap.add_argument("--only", nargs="+", default=None)
    args = ap.parse_args()

    refs = np.load(DCL.REFS_FILE)
    x_mean, x_std = bc_data.load_norm_stats()
    n = int(refs["n_mc"])
    mc_x0 = refs["mc_x0"]; mc_th = refs["mc_theta0"]
    mc_a = refs["mc_a"]; mc_e = refs["mc_e"]

    entries = [(l, c) for l, c in MANIFEST
               if args.only is None or l in args.only]

    header = (f"{'label':>9} {'rung':>4} {'n':>5} {'mismatch':>8} "
              f"{'mean_dev':>9} {'med_dev':>9} "
              f"{'mean_peak':>10} {'med_peak':>9}")
    print(header)
    print("-" * len(header))

    rows = []
    solver = None
    for rung in args.rungs:
        cfg = RL.LADDER[rung]
        # expert cache for this rung (built once, reused across networks)
        if solver is None:
            solver = EXP.build_solver()
        exp_paths = build_expert_cache(rung, cfg, refs, solver)

        for label, ckpt in entries:
            model, _, _ = DCL.load_checkpoint(ckpt)
            all_steps = []
            peaks = []
            mismatch = 0
            for i in range(n):
                tape = RL.DisturbanceTape(rung, i, cfg)
                netp = fly_network(model, mc_x0[i], float(mc_th[i]),
                                   float(mc_a[i]), float(mc_e[i]),
                                   x_mean, x_std, tape)
                expp = exp_paths[i]
                m = min(len(netp), len(expp))
                if len(netp) != len(expp):
                    mismatch += 1
                dev = np.linalg.norm(netp[:m] - expp[:m], axis=1)
                all_steps.append(dev)
                peaks.append(float(dev.max()))
            pooled = np.concatenate(all_steps)
            peaks = np.asarray(peaks)
            mean_dev = float(pooled.mean())
            med_dev = float(np.median(pooled))
            mean_peak = float(peaks.mean())
            med_peak = float(np.median(peaks))
            print(f"{label:>9} {rung:>4} {n:>5} {mismatch:>8} "
                  f"{mean_dev:>9.4f} {med_dev:>9.4f} "
                  f"{mean_peak:>10.4f} {med_peak:>9.4f}")
            rows.append([label, rung, n, mismatch, mean_dev, med_dev,
                         mean_peak, med_peak])

    with open("expert_path_deviation_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "rung", "n", "length_mismatch_pairs",
                    "mean_step_dev_m", "median_step_dev_m",
                    "mean_flight_peak_dev_m", "median_flight_peak_dev_m"])
        w.writerows(rows)
    print("\n  saved -> expert_path_deviation_summary.csv")


if __name__ == "__main__":
    main()